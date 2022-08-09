import torch
from torch.utils.data import TensorDataset
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

def Create_Datasets(dataPath,vars,tstPer):
	data = pd.read_pickle(dataPath) #Loads pandas data stored in pickle object
	y = np.array(data.loc[:,'Category']).astype(int)

	# Creates train test split from the data
	xTrn,xTst,yTrn,yTst = train_test_split(
		data.iloc[:,data.columns != "Category"],
		y,
		test_size=tstPer,  
		shuffle=True
		)

	#Convert to torch tensors
	xTrn_Torch = torch.Tensor(xTrn.loc[:,vars].to_numpy())
	xTst_Torch = torch.Tensor(xTst.loc[:,vars].to_numpy())
	yTrn_Torch = torch.Tensor(yTrn).type(torch.LongTensor)
	yTst_Torch = torch.Tensor(yTst).type(torch.LongTensor)

	# Creates a training dataset
	train_ds = torch.utils.data.TensorDataset(xTrn_Torch,yTrn_Torch)

	# Creates a testing dataset
	test_ds = torch.utils.data.TensorDataset(xTrn_Torch,yTrn_Torch)

	return train_ds,test_ds
