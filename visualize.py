from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image,ImageOps
import torchvision.transforms as transforms
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure

def read(index,size,start,end):
	path = './data/data/{}/'.format(index)
	inputs, labels = list(), list()
	transformation = transforms.Compose([transforms.Resize(size)])
	# for i in range(len(os.listdir(path+'img/'))):
	for i in range(start,end+1):
		# print(path+'img/'+'{}.png'.format(i))
		img = Image.open(path+'img/'+'{}.png'.format(i), mode='r').convert("L")
		gt = Image.open(path+'gt/'+'{}.png'.format(i), mode='r').convert("L")
		img = transformation(img)
		gt = transformation(gt)
		img = np.array(img.getdata()).reshape((size,size))
		img = (img-np.min(img))/(np.max(img)-np.min(img))
		inputs.append(img)
		labels.append(np.round(np.array(gt.getdata()).reshape((size,size))/255))
		# plt.imshow(inputs[-1].reshape((size,size)),cmap='gray')
		# plt.axis('off')
		# plt.show()
		# plt.imshow(labels[-1].reshape((size,size)),cmap='gray')
		# plt.axis('off')
		# plt.show()
	inputs = np.array(inputs).reshape((len(inputs),size,size))
	labels = np.array(labels).reshape((len(labels),size,size))
	return inputs, labels

def plot_3d(output,target,address,loss):
	threshold = 0 

	t = target.transpose(2,1,0)
	t_verts, t_faces, t_normals, t_values = measure.marching_cubes_lewiner(t, threshold)
	fig = plt.figure(figsize=(10, 10))
	ax = fig.add_subplot(111, projection='3d')
	t_mesh = Poly3DCollection(t_verts[t_faces], alpha=0.2)
	t_face_color = 'red'
	t_mesh.set_facecolor(t_face_color)

	p = output.transpose(2,1,0)
	verts, faces, normals, values = measure.marching_cubes_lewiner(p, threshold)
	mesh = Poly3DCollection(verts[faces], alpha=0.2)
	# face_color = [0.5, 0.5, 1]
	mesh.set_facecolor('blue')
	
	ax.add_collection3d(mesh)
	ax.add_collection3d(t_mesh)
	ax.set_xlim(0, p.shape[0])
	ax.set_ylim(0, p.shape[1])
	ax.set_zlim(0, p.shape[2])
	plt.title('Dice score: {}'.format(1-loss))
	fig.savefig("./data/inference_flexunet/{}_{}.png".format(round(1-loss,6),address))
	# plt.show()


if __name__ == '__main__':
	inputs, output = read(5,256,46,74)
	inputs, target = read(2,256,353,381)

	plot_3d(output,target)

