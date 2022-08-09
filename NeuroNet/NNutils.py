import torch
import torch.nn.utils.prune as prune
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from NN_DataLoad import Create_Datasets
import pdb

# Used to save the model weights to a backup file
def save_checkpoint(state,filename="my_checkpoint.pth.tar"):
	print("=> Saving Checkpoing")
	torch.save(state,filename)

# Used to load the learned weights of a model
def load_checkpoint(checkpoint,model):
	print("=> Loading Checkpoint ")
	model.load_state_dict(checkpoint["state_dict"])

def get_loaders(
	dataPath,
	vars,
	tstPer,
	batch_size,
):
	train_ds,test_ds = Create_Datasets(dataPath=dataPath,vars=vars,tstPer=tstPer)

	train_loader = torch.utils.data.DataLoader(train_ds,batch_size=batch_size,shuffle=True)
	test_loader = torch.utils.data.DataLoader(test_ds,batch_size=batch_size,shuffle=True)

	return train_loader,test_loader

def check_accuracy(loader,model,device="cuda"):
	num_correct = 0
	num_samples = 0

	# Disables gradient for testing
	with torch.no_grad():
		for x, y in loader:
			probs = model(x)
			predsInt = torch.nn.functional.softmax(probs,dim=1)
			v,preds = torch.max(predsInt,dim=1)

			num_correct += (preds==y).sum()
			num_samples += torch.numel(preds)

			print(
				f"Got {num_correct}/{num_samples} with acc {num_correct/num_samples*100:.2f}"
				)

			#pdb.set_trace()

