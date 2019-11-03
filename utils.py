import torch
import torch.nn.functional as F
import numpy as np
from torch import nn
import torch
import torch.optim as optim
from dataset import CTDataset,CTDataLoader
from flexible_model import dice_loss,Unet,Flex_Unet

def save_history(train_h,val_h,file_name):
	history = np.array([train_h,val_h])
	np.save(file_name,history)

def load_history(file_name):
	# print(np.load(file_name))
	return np.load(file_name)


def save_check_point(model,is_best,loss,epoch,optimizer):
	if is_best:
		state = {'epoch': epoch,
			 'best_loss': loss,
			 'model_state_dict': model.state_dict(),
			 'optimizer_state_dict': optimizer.state_dict()}
		filename = './model/Best_model_{}_{}.pth.tar'.format(epoch,round(loss,4))
		torch.save(state, filename)
		print("Model saved!")

def load_check_point(path):
	model = Flex_Unet(3,6)
	optimizer = optim.Adam(model.parameters())
	checkpoint = torch.load(path)
	model.load_state_dict(checkpoint['model_state_dict'])
	optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
	epoch = checkpoint['epoch']
	return model, optimizer,epoch