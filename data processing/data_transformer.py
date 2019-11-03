import numpy as np
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import os


# parameters
WW = 350
WL = 40
# lost: 9

for i in range(0,131):
	print("-----------------------{}---------------------".format(i))
	if not os.path.exists("../data/data/{}".format(i)) and os.path.exists("../raw data/lits/NII/volume/volume-{}.nii".format(i)): 
		os.mkdir("../data/data/{}".format(i))
		os.mkdir("../data/data/{}/img".format(i))
		os.mkdir("../data/data/{}/gt".format(i))

		img = nib.load("../raw data/lits/NII/volume/volume-{}.nii".format(i))
		gt = nib.load("../raw data/lits/NII/label/segmentation-{}.nii".format(i))
		header = img.header
		print(header["pixdim"])
		
		img, gt = np.array(img.dataobj).astype(float), np.array(gt.dataobj)
		# input rescaling
		# print(np.max(img),np.min(img),img.dtype)
		img[img>(WL+WW/2)] = np.nan
		img[img<(WL-WW/2)] = np.nan
		img[np.isnan(img)] = np.nanmin(img)
		img = (img-np.nanmin(img))/(np.nanmax(img)-np.nanmin(img))
		
		# label rescaling
		gt[gt==0] = 0
		gt[gt>0] = 1

		start_index_with_liver = -1
		end_index_with_liver = 0
		end_slices = False
		thickness = header["pixdim"][3]
		sample_mod = 5//thickness
		save_idx = 0
		for s in range(img.shape[-1]):
			g = gt[:,:,s]
			if np.sum(g)>0 and start_index_with_liver ==-1: start_index_with_liver = s
			if start_index_with_liver>-1 and np.sum(g) == 0 and not end_slices:  
				end_index_with_liver = s 
				end_slices = True
			# save
			if s%sample_mod == 0 and np.sum(g)>0:
				save_img = np.rot90(img[:,:,s])
				g = np.rot90(g)
				plt.imsave("../data/data/{}/img/{}.png".format(i,save_idx), save_img, cmap='gray')
				plt.imsave("../data/data/{}/gt/{}.png".format(i,save_idx), g, cmap='gray')
				save_idx+=1
		f = open("../data/data/{}/indice.txt".format(i), 'w', encoding = 'UTF-8')    # 也可使用指定路徑等方式，如： C:\A.txt
		f.write("{},{}".format(start_index_with_liver,end_index_with_liver-1))
		f.close()


	# plt.ion()
	# for s in range(im.shape[-1]):
	# 	save_img = np.rot90(im[:,:,s])
	# 	plt.imshow(save_img,cmap='gray')
	# 	plt.axis('off')
	# 	plt.pause(0.0000001)
	# plt.show()