import pandas as pd
import numpy as np
import os	# Used for opening files and managing paths
import matplotlib.pyplot as plt
from sklearn import (
	linear_model,
	metrics
	)
from sklearn.model_selection import train_test_split
import pdb
from itertools import combinations
from Utilities.utils import (
	CreateROC
	)

# Gets path to the files to be read
cwd = os.path.dirname(os.path.normpath(os.getcwd()))
print(cwd)
savePath = os.path.join(cwd,'DataPreparation','PD_Data_Prepped.pkl')
data = pd.read_pickle(savePath)

logreg = linear_model.LogisticRegression(C=1,solver='lbfgs',multi_class='multinomial')
y = np.array(data.loc[:,'Category']).astype(int)

xTrn,xTst,yTrn,yTst = train_test_split(data.iloc[:,data.columns != "Category"],y,test_size=0.5, random_state=1, shuffle=True)

#pdb.set_trace()

##___________________________________CLASSIFICATION USING SINGLE VARIABLE________________________________
testVars = ['BMPSITHT','BMPLEG','BMPARML','BMPBIAC','BMPBIIL']
cols = ['red','green','blue','gray','orange']
# Number of classes in the data
numClass = np.size(np.unique(y))

# Calculates rows and columns for subplots
nrow = np.ceil(np.power(numClass,1/2)).astype(int)
ncol = np.ceil(numClass/nrow).astype(int)
fig, ax_array = plt.subplots(nrows=nrow,ncols=ncol)
ax_array = ax_array.flatten() #Flattened to make iterating easier

for ic,cat in enumerate(testVars):
	xTrnNP = np.array(xTrn.loc[:,cat]).reshape(-1,1)
	logreg.fit(xTrnNP,yTrn)

	#Training Accuracy
	yTrnPred = logreg.predict(xTrnNP)
	trnAcc = np.sum(yTrnPred==yTrn)/np.size(yTrn)
	print(cat,"Training accuracy is: ", trnAcc)

	#Testing Accuracy
	xTstNP = np.array(xTst.loc[:,cat]).reshape(-1,1)
	yTstPred = logreg.predict(xTstNP)
	tstAcc = np.sum(yTstPred==yTst)/np.size(yTst)
	print(cat,"Testing accuracy is: ",tstAcc)

	#Confusion Matrix
	#c_mat = metrics.confusion_matrix(yTst,yTstPred,normalize='true') #Create confusion matrix
	#cm_disp = metrics.ConfusionMatrixDisplay(confusion_matrix=c_mat, display_labels = ['0','1','2'])
	#cm_disp.plot()
	#cm_disp.ax_.set_title(cat) #Adds title

	#Calculates the probability of an object being in a given class
	probTst = logreg.predict_proba(xTstNP)
	CreateROC(probTst,cols[ic],cat,ax_array,yTst)
plt.show()

##___________________________________CLASSIFICATION USING DOUBLE VARIABLE________________________________
# Calculates rows and columns for subplots
nrow = np.ceil(np.power(numClass,1/2)).astype(int)
ncol = np.ceil(numClass/nrow).astype(int)
fig, ax_array1 = plt.subplots(nrows=nrow,ncols=ncol)
ax_array1 = ax_array1.flatten() #Flattened to make iterating easier
testVars = ['BMPSITHT','BMPLEG','BMPARML','BMPBIAC']
cols = ['red','green','blue','gray','orange','brown']

for iC,comb in enumerate(combinations(testVars,2)):
	xTrnNP = np.array(xTrn.loc[:,comb])
	logreg.fit(xTrnNP,yTrn)

	#Training Accuracy
	yTrnPred = logreg.predict(xTrnNP)
	trnAcc = np.sum(yTrnPred==yTrn)/np.size(yTrn)
	print(comb,"Training accuracy is: ", trnAcc)

	#Testing Accuracy
	xTstNP = np.array(xTst.loc[:,comb])
	yTstPred = logreg.predict(xTstNP)
	tstAcc = np.sum(yTstPred==yTst)/np.size(yTst)
	print(comb,"Testing accuracy is: ",tstAcc)

	#Confusion Matrix
	#c_mat = metrics.confusion_matrix(yTst,yTstPred,normalize='true') #Create confusion matrix
	#cm_disp = metrics.ConfusionMatrixDisplay(confusion_matrix=c_mat, display_labels = ['0','1','2'])
	#cm_disp.plot()
	#cm_disp.ax_.set_title(comb) #Adds title

	#Calculates the probability of an object being in a given class
	probTst = logreg.predict_proba(xTstNP)
	CreateROC(probTst,cols[iC],comb,ax_array1,yTst)

plt.show()

##___________________________________CLASSIFICATION USING TRIPLE VARIABLE________________________________
# Calculates rows and columns for subplots
nrow = np.ceil(np.power(numClass,1/2)).astype(int)
ncol = np.ceil(numClass/nrow).astype(int)
fig, ax_array1 = plt.subplots(nrows=nrow,ncols=ncol)
ax_array1 = ax_array1.flatten() #Flattened to make iterating easier
testVars = ['BMPSITHT','BMPLEG','BMPARML','BMPBIAC']

for iC,comb in enumerate(combinations(testVars,3)):
	xTrnNP = np.array(xTrn.loc[:,comb])
	logreg.fit(xTrnNP,yTrn)

	#Training Accuracy
	yTrnPred = logreg.predict(xTrnNP)
	trnAcc = np.sum(yTrnPred==yTrn)/np.size(yTrn)
	print(comb,"Training accuracy is: ", trnAcc)

	#Testing Accuracy
	xTstNP = np.array(xTst.loc[:,comb])
	yTstPred = logreg.predict(xTstNP)
	tstAcc = np.sum(yTstPred==yTst)/np.size(yTst)
	print(comb,"Testing accuracy is: ",tstAcc)

	#Confusion Matrix
	c_mat = metrics.confusion_matrix(yTst,yTstPred,normalize='true') #Create confusion matrix
	cm_disp = metrics.ConfusionMatrixDisplay(confusion_matrix=c_mat, display_labels = ['0','1','2'])
	cm_disp.plot()
	cm_disp.ax_.set_title(comb) #Adds title

	#Calculates the probability of an object being in a given class
	probTst = logreg.predict_proba(xTstNP)
	CreateROC(probTst,cols[iC],comb,ax_array1,yTst)
plt.show()



