import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.misc
import dicom
import os

def padding_calculator(shape):
	H,W = shape
	up_pad, down_pad, left_pad, right_pad = 0,0,0,0
	if H != 512:
		up_pad = down_pad = (512-H)//2
		up_pad += (512-H)%2
	if W != 512:
		left_pad = right_pad = (512-W)//2
		left_pad += (512-W)%2
	return up_pad, down_pad, left_pad, right_pad 

root_folder_path = '../raw data/2019Sep/'

cases_path = [root_folder_path+p+"/"+os.listdir(root_folder_path+p)[0]+"/" for p in os.listdir(root_folder_path)]
print(len(os.listdir(root_folder_path)))

for i,c in enumerate(cases_path):
	print("==================={}====================".format(i)) 
	print(c)
	folder_list = os.listdir(c)
	make_folder_name = c.split("/")[-3]
	# if i==0:
	if 49<=i<=57:
		folder_list = os.listdir("/".join(c.split('/')[:-2])+"/")
		c = "/".join(c.split('/')[:-2])+"/"
	dcm_folder = c+list(filter(lambda x:x[0:3]=="C+P", folder_list))[0]+"/"
	target_folder = c+list(filter(lambda x:x[0:9]=="C+P Liver", folder_list))[0]+"/"
	if not os.path.exists("../data/2019Sep/"+make_folder_name): 
		os.mkdir("../data/2019Sep/"+make_folder_name)
		os.mkdir("../data/2019Sep/"+make_folder_name+"/img")
		os.mkdir("../data/2019Sep/"+make_folder_name+"/gt")

	n1 = len(os.listdir(dcm_folder))
	n2 = len(os.listdir(target_folder))
	if n1!=n2: print(n1,n2)
	dcm = [dcm_folder+d for d in os.listdir(dcm_folder)]
	target = [target_folder+d for d in os.listdir(target_folder)]
	img_ouput_path = "../data/2019Sep/"+make_folder_name+"/img/"
	target_ouptut_path = "../data/2019Sep/"+make_folder_name+"/gt/"
	save_idx = 0
	flag = 1
	for d_path,t_path in zip(dcm[::-1],target[::-1]):

		# img rescaling
		ds = dicom.read_file(d_path) 
		intercept = ds.RescaleIntercept
		slope = ds.RescaleSlope
		WL = ds['0028','1050'].value
		WW = ds['0028','1051'].value
		im = ds.pixel_array
		im = im*slope+intercept			
		im = im.astype(np.float32)

		im[im>(WL+WW/2)] = np.nan
		im[im<(WL-WW/2)] = np.nan
		im[np.isnan(im)] = np.nanmin(im)
		im = (im-np.nanmin(im))/(np.nanmax(im)-np.nanmin(im))

		# target rescaling
		gt = dicom.read_file(t_path) 
		gt = gt.pixel_array
		gt = gt.astype(np.float32)
		gt[gt==0] = 0
		gt[gt>0] = 1
		if flag:
			print("Check shape",im.shape,gt.shape)
			flag = 0
		if np.sum(gt)>100:
			if im.shape[0]<512 or im.shape[1]<512 or gt.shape[0]<512 or gt.shape[1]<512:
				# print("pad",im.shape,gt.shape)
				up_pad, down_pad, left_pad, right_pad = padding_calculator(im.shape)
				im = np.pad(im,((up_pad, down_pad),(left_pad, right_pad)),mode="constant",constant_values=0)
				up_pad, down_pad, left_pad, right_pad = padding_calculator(gt.shape)
				gt = np.pad(gt,((up_pad, down_pad),(left_pad, right_pad)),mode="constant",constant_values=0)
				# print("done",im.shape,gt.shape)
			scipy.misc.imsave(img_ouput_path+str(save_idx)+'.png', im)
			scipy.misc.imsave(target_ouptut_path+str(save_idx)+'.png', gt)
			save_idx+=1
	flag = 1
			# plt.imshow(im,'gray')
			# plt.title(d_path)
			# plt.show()
			# plt.imshow(gt,'gray')
			# plt.title(t_path)
			# plt.show()
			




"""

for i in range(11):
# for i in [1,16,21]:
	index = "%03d" % (i)
	folder_path = data_path+str(i)+'.dcm'
	if os.path.exists(folder_path) :
		img_path = folder_path
		
		num+=1
		
		ds = dicom.read_file(img_path) 
		intercept = ds.RescaleIntercept
		slope = ds.RescaleSlope
		print(i)

		WL = ds['0028','1050'].value
		WW = ds['0028','1051'].value
		# print(WL,WW)
		# print(slope,intercept)
		im = ds.pixel_array
		im = im*slope+intercept
		
		im = im.astype(np.float32)

		im[im>(WL+WW/2)] = np.nan
		im[im<(WL-WW/2)] = np.nan
		# print(im)
		im[np.isnan(im)] = np.nanmin(im)
		im = (im-np.nanmin(im))/(np.nanmax(im)-np.nanmin(im))


		# plt.imshow(im,'gray')
		# plt.title(str(index))
		# plt.show()
		# plt.imshow(target,'gray')
		# plt.title(str(index))
		# plt.show()
		scipy.misc.imsave(output_path+str(i)+'.png', im)
		"""