import pandas as pd
import numpy as np
import os	# Used for opening files and managing paths
import matplotlib.pyplot as plt
from Utilities.utils import (
	Remove_BlankButAcceptable,
	SplitIntoCategorys,
	BalanceData,
	StandardizeData
	)


cwd = os.getcwd()
savePath = os.path.join(cwd,'PD_Data.pkl')

binsH = np.arange(70,210,2)
data = pd.read_pickle(savePath)

#Removes the 88888 values from the data (see utils.py)
data = Remove_BlankButAcceptable(data)

#Plots a histogram of the height data
fig1, (ax1) = plt.subplots(nrows=1, ncols=1)
ax1.hist(data['BMPHT'],bins=binsH)
ax1.set_xlabel('Height (cm)')
ax1.set_ylabel('Occurences')

#Adds categorys to the data
splitIn = np.array([0,44,70])
splitCm = splitIn*2.54
print(splitCm)
for i in splitCm[1::]:
	ax1.plot([i,i],[0,ax1.get_ylim()[1]],'k-')

data = SplitIntoCategorys(data,splitCm)

binsC=[0,1,2,3]
fig2, (ax2) = plt.subplots(nrows=1, ncols=1)
ax2.hist(data['Category'],bins=binsC,align='left')
ax2.set_xlabel('Classes')
ax2.set_ylabel('Occurences')
ax2.set_xticks([0,1,2])

#Balances the data
balData = BalanceData(data)
fig3, (ax3) = plt.subplots(nrows=1, ncols=1)
ax3.hist(balData['BMPHT'],bins=binsH)
ax3.set_xlabel('Height (cm)')
ax3.set_ylabel('Occurences')
fig4, (ax4) = plt.subplots(nrows=1, ncols=1)
ax4.hist(balData['Category'],bins=binsC,align='left')
ax4.set_xlabel('Classes')
ax4.set_ylabel('Occurences')
ax4.set_xticks([0,1,2])

plt.show()

#Standardize the data
colStand = ['BMPWT','BMPSITHT','BMPLEG','BMPARML','BMPBIAC','BMPBIIL']
stanData = StandardizeData(balData,colStand)

# Gets path to the files to be read
cwd = os.getcwd()
savePath = os.path.join(cwd,'PD_Data_Prepped.pkl')
stanData.to_pickle(savePath)