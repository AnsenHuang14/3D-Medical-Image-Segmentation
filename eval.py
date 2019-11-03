import torch.nn.functional as F
import numpy as np
from torch import nn
import torch
import torch.optim as optim
from dataset import CTDataset,CTDataLoader
from flexible_model import dice_loss,Unet, Flex_Unet
from utils import save_check_point, load_check_point, save_history, load_history
from tqdm import tqdm
from visualize import plot_3d
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = './model/Best_model_14_0.0798.pth.tar'
batch_size = 1

# Load model checkpoint that is to be evaluated
model,_,_ = load_check_point(checkpoint)
model.double().to(device)
loss_function = dice_loss
model.eval()
# build data loader
data = CTDataset('./data/DCM_Test.json')
test_loader = CTDataLoader(data,1,batch_size=batch_size,mode="testing")

# evaluate
result = []
area = []
with torch.no_grad():
	for i, (data, target, address) in enumerate(tqdm(test_loader, desc='Evaluating')):
		output = model(data.to(device))
		p, t = np.round(np.array(output.cpu())), np.round(np.array(target.cpu()))
		loss = loss_function(output.to(device), target.to(device))
		plot_3d(p.reshape(p.shape[2:]),t.reshape(t.shape[2:]),address,loss.item())
		# print(np.sum(p),np.sum(t))
		area.append(np.sum(p)/np.sum(t))
		result.append(loss.item())
		
	print ("==================== Dice loss:", np.mean(result),", Std, min, max :", np.std(result), np.min(result), np.max(result), "======================")
	print ("==================== Area ratio (p/t):", np.mean(area),", Std, min, max:", np.std(area), np.min(area), np.max(area), "======================")
