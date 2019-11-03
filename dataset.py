import json
import os
from PIL import Image,ImageOps
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as transforms
import torch
import numpy as np
import matplotlib.pyplot as plt

class CTDataset(Dataset):
	def __init__(self, file_name):
		with open(os.path.join(file_name), 'r') as j:
			self.instance = json.load(j)
	def __getitem__(self, i):
		# print(self.instance[i]['id'],self.instance[i]['corr'] )
		inputs, labels = list(), list()
		resize = size =  (256, 256)
		transformation = transforms.Compose([transforms.Resize(size=resize)])
		size = 256
		# print(self.instance[i]['address'])
		for index in range(self.instance[i]['number_of_instance']):
			img = Image.open(self.instance[i]['address']+'img/'+'{}.png'.format(index), mode='r')
			gt = Image.open(self.instance[i]['address']+'gt/'+'{}.png'.format(index), mode='r')
			img = transformation(img.convert("L"))
			gt = transformation(gt.convert("L"))

			img = np.array(img.getdata())
			gt = np.array(gt.getdata())
			
			img = img.reshape((size,size))
			img = (img-np.min(img))/(np.max(img)-np.min(img))
			

			inputs.append(img)
			labels.append(np.round(gt.reshape((size,size))/255))
			# print(np.min(img),np.max(img))
			# plt.imshow(inputs[-1].reshape((size,size)),cmap='gray')
			# plt.axis('off')
			# plt.show()
			# plt.imshow(labels[-1].reshape((size,size)),cmap='gray')
			# plt.axis('off')
			# plt.show()
		inputs = np.array(inputs).reshape((1,len(inputs),size,size))
		labels =  np.array(labels).reshape((1,len(labels),size,size))
		
		return inputs,labels,self.instance[i]['address'].split("/")[-2]
	def __len__(self):
		return len(self.instance)

def CTDataLoader(Dset,data_split_ratio,batch_size,mode):
	## create training and validation split 
	print("Data loader mode:",mode)
	if mode=="training": 
		split = int(data_split_ratio * len(Dset))
		index_list = list(range(len(Dset)))
		train_idx, valid_idx = index_list[:split], index_list[split:]
		print("Number of training/validation samples:",len(index_list[:split]),"/",len(index_list[split:]))
		## create sampler objects using SubsetRandomSampler
		tr_sampler = SubsetRandomSampler(train_idx)
		val_sampler = SubsetRandomSampler(valid_idx)
		## create iterator objects for train and valid datasets
		trainloader = DataLoader(Dset, batch_size=batch_size, shuffle=False, sampler=tr_sampler)
		validloader = DataLoader(Dset, batch_size=batch_size, shuffle=False, sampler=val_sampler)
		return trainloader,validloader
	elif mode=="testing":
		print("Number of testing samples:",len(Dset))
		testloader = DataLoader(Dset, batch_size=batch_size, shuffle=False)
		return testloader
	else:
		print("Mode: training/testing")

if __name__ == '__main__':
	data = CTDataset('./data/Test.json')
	test_loader= CTDataLoader(data,0.75,batch_size=1,mode="testing")
	for idx,(i,t) in enumerate(test_loader):
		print(i.shape,t.shape)
		
		

