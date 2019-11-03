from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os

# method – One of PIL.Image.FLIP_LEFT_RIGHT, PIL.Image.FLIP_TOP_BOTTOM, PIL.Image.ROTATE_90, 
# PIL.Image.ROTATE_180, PIL.Image.ROTATE_270 or PIL.Image.TRANSPOSE.
for i in range(68,80):
	if os.path.exists("../raw data/lits/NII/volume/volume-{}.nii".format(i)):
		print('----------',i,'-----------') 
		number_of_instance = len(os.listdir('../data/data/{}/gt/'.format(i)))
		for idx in range(number_of_instance):
			# idx = 10
			image = Image.open('../data/data/{}/img/{}.png'.format(i,idx), mode='r')
			image = image.convert("L")
			gt = Image.open('../data/data/{}/gt/{}.png'.format(i,idx), mode='r')
			gt = gt.convert("L")
			# image = image.transpose(Image.FLIP_TOP_BOTTOM)
			# gt = gt.transpose(Image.FLIP_TOP_BOTTOM)
			image = image.transpose(Image.FLIP_LEFT_RIGHT)
			gt = gt.transpose(Image.FLIP_LEFT_RIGHT)
			# plt.imshow(np.array(image.getdata()).reshape((512,512)),cmap='gray')
			# plt.axis('off')
			# plt.show()
			# plt.imshow(np.array(gt.getdata()).reshape((512,512)),cmap='gray')
			# plt.axis('off')
			# plt.show()
			image.save('../data/data/{}/img/{}.png'.format(i,idx))
			gt.save('../data/data/{}/gt/{}.png'.format(i,idx))
		


