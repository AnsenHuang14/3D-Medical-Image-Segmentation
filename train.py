import torch.nn.functional as F
import numpy as np
from torch import nn
import torch
import torch.optim as optim
from dataset import CTDataset,CTDataLoader
from flexible_model import dice_loss,Unet, Flex_Unet
from utils import save_check_point, load_check_point, save_history, load_history
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# parameters
best_loss = 100
early_stop = 5
not_improve = 0
batch_size = 1
down_scale = 3
first_ch = 6

# data loader
data = CTDataset('./data/Train.json')
train_loader, val_loader = CTDataLoader(data,0.75,batch_size=batch_size,mode="training")

# model, loss function, optimizer initialization
model = Flex_Unet(down_scale, first_ch)
optimizer = optim.Adam(model.parameters())

# checkpoint_path = ''
# model, optimizer, start_epoch = load_check_point(checkpoint_path)
print("Number of paras:",sum(p.numel() for p in model.parameters() if p.requires_grad))
model.double().to(device)
loss_function = dice_loss

train_history, val_history = [],[]
for epoch in range(1, 1000):
	print('-----------','epoch:',epoch,'-----------')
	train_loss, valid_loss = [], [] 
	# training part
	model.train()
	for data, target, _ in train_loader:
		optimizer.zero_grad()
		output = model(data.to(device))
		loss = loss_function(output.to(device), target.to(device))
		loss.backward()
		optimizer.step()
		train_loss.append(loss.item()) 

	# evaluation part
	with torch.no_grad(): 
		model.eval()
		for data, target,_ in val_loader:
			output = model(data.to(device))
			loss = loss_function(output.to(device), target.to(device))
			valid_loss.append(loss.item())

	print ("Epoch:", epoch, "Training Loss: ", np.mean(train_loss), "Valid Loss: ", np.mean(valid_loss))

	train_history.append(np.mean(train_loss))
	val_history.append(np.mean(valid_loss))

	# save model if performance improved
	is_best = np.mean(valid_loss) < best_loss
	best_loss = min(np.mean(valid_loss), best_loss)
	# record number of no improve epochs for early stopping
	if not is_best: not_improve+=1
	else: not_improve=0
	if not_improve>early_stop: 
		print("======== Early stop at epoch ",epoch,"==========")
		break 
	save_check_point(model,is_best,best_loss,epoch,optimizer)
print("train history:",len(train_history),",validate history:",len(val_history))
save_history(train_history,val_history,"./model/Best_model_history.npy")