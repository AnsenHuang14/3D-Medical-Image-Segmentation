import numpy as np
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import os

a = set()
for i in range(131):
	if os.path.exists("../raw data/lits/NII/volume/volume-{}.nii".format(i)):
		img = nib.load("../raw data/lits/NII/volume/volume-{}.nii".format(i))
		gt = nib.load("../raw data/lits/NII/label/segmentation-{}.nii".format(i))
		header = img.header
		thickness = header["pixdim"][3]
		a.add(thickness)
		print(i,thickness,img.shape[-1],5//thickness)
		
print(a)