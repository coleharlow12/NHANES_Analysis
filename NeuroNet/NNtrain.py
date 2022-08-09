import torch
import torch.nn as nn
import torch.optim as optim
from model import Dense
from tqdm import tqdm
from NNutils import (
	load_checkpoint,
	save_checkpoint,
	check_accuracy,
	get_loaders,
	)
import os

LEARNING_RATE = 1e-3
BATCH_SIZE = 500
NUM_EPOCHS = 100
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Gets path to the files to be read
cwd = os.path.dirname(os.path.normpath(os.getcwd()))
print(cwd)
savePath = os.path.join(cwd,'DataPreparation','PD_Data_Prepped.pkl')

DATAPATH = savePath

# Does one epoch of training
def train_fn(loader,model,optimizer, loss_fn, scaler):
	loop = tqdm(loader) #Creates progress bar

	for batch_idx, (data,targets) in enumerate(loop):

		#Forward 
		predictions = model(data) #Predicts data output is N x Classes
		loss = loss_fn(predictions, targets) #Expects the input to have dimension N x Classes and be unnormalized

		#Backwards
		optimizer.zero_grad() #Sets gradients to zero
		scaler.scale(loss).backward()
		scaler.step(optimizer)
		scaler.update()

		# Update tqdm loop
		loop.set_postfix(loss=loss.item())

# Trains the model
def main():
	model = Dense(inChannels1=3,outChannels1=10,outChannels2=10,outChannels3=3)
	loss_fn = nn.CrossEntropyLoss()
	scaler = torch.cuda.amp.GradScaler()
	optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

	train_loader, test_loader = get_loaders(
		dataPath = DATAPATH,
		vars = ['BMPSITHT','BMPLEG','BMPARML'],
		tstPer = 0.5,
		batch_size = BATCH_SIZE
	)

	for epoch in range(NUM_EPOCHS):
		train_fn(train_loader, model, optimizer, loss_fn, scaler)

		# Save Model
		checkpoint = {
			"state_dict":model.state_dict(),
			"optimizer":optimizer.state_dict(),
		}
		save_checkpoint(checkpoint)

		# Check Accuracy
		check_accuracy(test_loader, model, device=DEVICE)


if __name__ == "__main__":
	main()


