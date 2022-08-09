import torch
import torch.nn as nn

class Dense(nn.Module):
	#Runs when class is initialized
	def __init__(self,inChannels1,outChannels1,outChannels2,outChannels3):
		
		super(Dense, self).__init__()
		self.dens = nn.Sequential(
			nn.Linear(
				in_features=inChannels1,
				out_features=outChannels1,
				bias=True
				),
			nn.BatchNorm1d(num_features=outChannels1),
			nn.ReLU(inplace=True),
			nn.Linear(
				in_features=outChannels1,
				out_features=outChannels2,
				bias=True
				),
			nn.BatchNorm1d(num_features=outChannels2),
			nn.ReLU(inplace=True), 
			nn.Linear(
				in_features=outChannels2,
				out_features=outChannels3,
				bias=True
				),
			nn.BatchNorm1d(num_features=outChannels3),
			nn.ReLU(inplace=True)
			)

	# Defines the forward step through the neural network
	def forward(self,x):
		x = self.dens(x)
		return x


