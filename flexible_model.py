import torch.nn.functional as F
import numpy as np
from torch import nn
import torch
import matplotlib.pyplot as plt


def dice_loss(output, target):
	smooth = 1.
	intersect = torch.sum(output * target)
	union = torch.sum(output+target)
	return 1.-(2.*intersect+smooth)/(union+smooth)

class down_block(nn.Module): # input: ch / output:  2*ch
	def __init__(self, input_channel):
		super(down_block, self).__init__()
		self.conv1 = nn.Conv3d(input_channel, input_channel, kernel_size=3, groups=1 ,padding=1, stride=1)
		self.conv2 = nn.Conv3d(input_channel, 2*input_channel, kernel_size=3, groups=1 ,padding=1, stride=1)
		self.BN1 = nn.BatchNorm3d(input_channel)
		self.BN2 = nn.BatchNorm3d(2*input_channel)
	def forward(self, x):
		x = self.conv1(x)
		x = F.relu(self.BN1(x))
		x = self.conv2(x)
		x = F.relu(self.BN2(x))
		return x

class block(nn.Module): # input: ch / output: ch
	def __init__(self, input_channel, output_channel):
		super(block, self).__init__()
		self.conv1 = nn.Conv3d(input_channel, output_channel, kernel_size=3, groups=1 ,padding=1, stride=1)
		self.conv2 = nn.Conv3d(output_channel, output_channel, kernel_size=3, groups=1 ,padding=1, stride=1)
		self.BN1 = nn.BatchNorm3d(output_channel)
		self.BN2 = nn.BatchNorm3d(output_channel)
	def forward(self, x):
		x = self.conv1(x)
		x = F.relu(self.BN1(x))
		x = self.conv2(x)
		x = F.relu(self.BN2(x))
		return x

class Unet(nn.Module):
	def __init__(self):
		super(Unet, self).__init__() # batch,channel,D,H,W
		# input shape: 1, 1, D, 512, 512
		self.down_block1 = down_block(1)       # 4, 256
		self.down_block2 = down_block(4)       # 16, 128
		self.down_block3 = down_block(16)      # 64, 64
		
		self.maxpool = nn.MaxPool3d(kernel_size=2, stride=2, padding=0, dilation=1, return_indices=False, ceil_mode=False)
		
		self.conv1 = nn.Conv3d(64, 64, kernel_size=3, groups=1 ,padding=1, stride=1)
		self.conv2 = nn.Conv3d(64, 64, kernel_size=3, groups=1 ,padding=1, stride=1)
		self.BN1 = nn.BatchNorm3d(64)
		self.BN2 = nn.BatchNorm3d(64)
		self.upconv1 = nn.ConvTranspose3d(64, 32, kernel_size=2 , stride=2, padding=0, output_padding=0, groups=1, bias=True, dilation=1)		
		# 32, 128

		self.up_block1 = up_block(16+32)       
		self.upconv2 = nn.ConvTranspose3d(48, 24, kernel_size=2 , stride=2, padding=0, output_padding=0, groups=1, bias=True, dilation=1)		
		# 24, 256
		
		self.up_block2 = up_block(24+4)
		self.upconv3 = nn.ConvTranspose3d(28, 16, kernel_size=2 , stride=2, padding=0, output_padding=0, groups=1, bias=True, dilation=1)		
		# 16, 512

		self.conv3 = nn.Conv3d(16+4, 8, kernel_size=3, groups=1 ,padding=1, stride=1)
		self.conv4 = nn.Conv3d(8, 1, kernel_size=3, groups=1 ,padding=1, stride=1)
		self.BN3 = nn.BatchNorm3d(8)
		self.BN4 = nn.BatchNorm3d(1)

	def forward(self, x):
		block1_x = self.down_block1(x) # 4,512
		block1_mp_x = self.maxpool(block1_x) # 4,256

		block2_x = self.down_block2(block1_mp_x) # 16,256
		block2_mp_x = self.maxpool(block2_x) # 16,128

		block3_x = self.down_block3(block2_mp_x) # 64,128
		block3_mp_x = self.maxpool(block3_x) # 64,64

		x = self.conv1(block3_mp_x) # 64,64
		x = F.relu(self.BN1(x))
		x = self.conv2(x) # 64,64
		x = F.relu(self.BN2(x))

		x = self.upconv1(x) # 32, 128
		x = F.pad(x,(0,0,0,0,0,block2_mp_x.size(2)-x.size(2)),mode='constant',value=0) # 32, 128

		x = torch.cat([x,block2_mp_x],1) # 32+16, 128
		x = self.up_block1(x) # 48, 128
		
		x = self.upconv2(x) # 24, 256
		x = F.pad(x,(0,0,0,0,0,block1_mp_x.size(2)-x.size(2)),mode='constant',value=0) # 24, 256
		
		x = torch.cat([x,block1_mp_x],1) # 24+4, 256
		x = self.upconv3(x) # 16, 512

		x = F.pad(x,(0,0,0,0,0,block1_x.size(2)-x.size(2)),mode='constant',value=0) # 16, 512
		x = torch.cat([x,block1_x],1) # 20, 512

		x = self.conv3(x) # 8,512
		x = F.relu(self.BN3(x))
		x = self.conv4(x) # 1,512
		x = torch.sigmoid(self.BN4(x))

		return x


class Flex_Unet(nn.Module):
	def __init__(self, down_scale, first_channel=32):
		super(Flex_Unet, self).__init__() # shape: batch, channel, D, H, W
		self.down_scale = down_scale # down sampling scale
		down_ch_out = [first_channel*2**(x+1) for x in range(down_scale+1)]
		upconv_ch_in_out = down_ch_out[:0:-1]
		cat_out = [x+y for x,y in zip(upconv_ch_in_out, down_ch_out[-2::-1])]
		up_ch_out = down_ch_out[-2::-1]
		print("down_ch_out",down_ch_out)
		print("upconv_ch_in_out",upconv_ch_in_out)
		print("cat_out",cat_out)
		print("up_ch_out",up_ch_out)

		module = {"first_conv" : nn.Conv3d(1, first_channel, kernel_size=3, groups=1 ,padding=1, stride=1),
				"maxpool" : nn.MaxPool3d(kernel_size=2, stride=2, padding=0, dilation=1, return_indices=False, ceil_mode=False)}
		self.layers = nn.ModuleDict(module)
		for i in range(down_scale+1):
			self.layers.update({"down_conv{}".format(i) : down_block(down_ch_out[i]//2)})
			if i < down_scale:
				self.layers.update({"up_conv{}".format(i) : nn.ConvTranspose3d(upconv_ch_in_out[i], upconv_ch_in_out[i], kernel_size=2 , stride=2, padding=0, output_padding=0, groups=1, bias=True, dilation=1)})
				self.layers.update({"up_block{}".format(i) : block(cat_out[i],up_ch_out[i])})
		self.layers.update({"output_conv" :  nn.Conv3d(up_ch_out[-1], 1, kernel_size=3, groups=1 ,padding=1, stride=1) })

		# print(self.layers)
		# # print(self.layers)
		# for n,l in self.layers.items():
		# 	print(n,l)
			# print(self.layers[n])
		
	def forward(self, x):
		output = {}
		x = self.layers["first_conv"](x)
		for i in range(self.down_scale+1): 
			name = "down_conv{}".format(i)
			output.update({ name : self.layers[name](x)})
			if i<self.down_scale:
				x = self.layers["maxpool"](output["down_conv{}".format(i)])
			else:
				x = self.layers[name](x)
		
		for i in range(self.down_scale):
			# print(i,x.size())
			upconv_name = "up_conv{}".format(i)
			upblock_name = "up_block{}".format(i)
			cat_name = "down_conv{}".format(self.down_scale-1-i)
			x = self.layers[upconv_name](x)
			# print(upconv_name, x.size())
			# print(cat_name, output[cat_name].size())
			if output[cat_name].size(2)-x.size(2)>0:
				x = F.pad(x,(0,0,0,0,0,output[cat_name].size(2)-x.size(2)),mode='constant',value=0)
			x = torch.cat([x,output[cat_name]],1)
			del output[cat_name]
			x = self.layers[upblock_name](x)
			
		x = self.layers["output_conv"](x)
		x = torch.sigmoid(x)
		return x

if __name__ == '__main__':
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model = Flex_Unet(4,16)
	# print("--------------parameters--------------")
	# for param in model.parameters():
	#   print(param.data.size())
	# print("--------------------------------------")
	inputs = torch.randn(1, 1, 20, 64, 64)
	target = torch.randn(1, 1, 20, 64, 64)
	
	print(dice_loss(inputs,target))
	output = model(inputs.to(device))
	print(dice_loss(output,target))

	# print("output size:",output.size())



"""
class ConvLstmCell(nn.Module):
	def __init__(self,input_channel,output_channel):
		super(ConvLstmCell, self).__init__()
		## define the layers
		self.igate_xt_conv = nn.Conv2d(input_channel, output_channel, 3, padding=1, stride=1)
		self.igate_ht_conv = nn.Conv2d(input_channel, output_channel, 3, padding=1, stride=1)
		self.igate_ct_conv = nn.Conv2d(input_channel, output_channel, 3, padding=1, stride=1)

		self.fgate_xt_conv = nn.Conv2d(input_channel, output_channel, 3, padding=1, stride=1)
		self.fgate_ht_conv = nn.Conv2d(input_channel, output_channel, 3, padding=1, stride=1)
		self.fgate_ct_conv = nn.Conv2d(input_channel, output_channel, 3, padding=1, stride=1)

		self.ogate_xt_conv = nn.Conv2d(input_channel, output_channel, 3, padding=1, stride=1)
		self.ogate_ht_conv = nn.Conv2d(input_channel, output_channel, 3, padding=1, stride=1)
		self.ogate_ct_conv = nn.Conv2d(input_channel, output_channel, 3, padding=1, stride=1)

		
	def forward(self, x):
		batch, num_slices, height, width = x.size
		output_list = []
		for t in range(num_slices):
			if t == 0: hidden_state = zero_tensor
			input_gate = 
			forget_gate = 
			output_gate = 

		x = self.BN1(self.pool(F.relu(self.conv1(x))))
		x = self.BN2(self.pool(F.relu(self.conv2(x))))
		# for i in range(x.size(1)):
		# 	plt.imshow(x.cpu().detach().numpy()[0,i,:,:],cmap='gray')
		# 	plt.show()
		x = self.BN3(self.pool(F.relu(self.conv3(x))))
		x = self.BN4(self.pool(F.relu(self.conv4(x))))
		x = F.relu(self.lin_conv1(x))
		x = self.lin_conv2(x)
		return torch.tanh(x).view(x.size(0))
"""