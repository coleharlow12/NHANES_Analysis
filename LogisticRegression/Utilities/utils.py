import pandas as pd
import numpy as np

def Remove_BlankButAcceptable(df):
	#Some of the values in the dataset are given the value 8888
	#According to the documentation 8888 implies the data is 
	#Blank but Applicable. I am not sure what it means but I am
	#Removing all entries with the value 8888 as they may alter results

	df[df>1000] = np.nan
	newDF = df.dropna().copy(deep=True)
	newDF.reset_index()

	return(newDF)

def SplitIntoCategorys(df,split):
	'''
	In order to do training I need to split the data into categories based on
	their heights. To do this a df is entered which must include a column with
	label BMPHT. Additionally a numpy array with entries to split the data into
	is also input. The function then outputs the dataframe with a new column of
	categories
	'''
	df["Category"] = " "
	for i, val in enumerate(split):
		print(val)
		isTrue = df.index[df["Height"]>=val].to_list()
		print(type(isTrue))
		df.loc[isTrue,"Category"] = i

	return df.copy(deep=True)

def BalanceData(df):
	'''
	Balances the data so that each category is equally represented in the
	dataset. 
		df: Dataframe with a column "Category"
	'''
	Cats = pd.unique(df['Category'])
	numCats = len(Cats)
	numOcc = np.zeros([numCats,1])

	catInd = {} #Stores a list for each category of the indices with that value
	for i,cat in enumerate(Cats):
		catInd[str(cat)] = df.index[df['Category']==cat].to_list()
		numOcc[i] = len(catInd[str(cat)])

	minOcc = numOcc.min()

	indKeep = []
	for i,cat in enumerate(Cats):
		if numOcc[i] is not minOcc:
			indArr = np.array(catInd[str(cat)])
			keepArr = np.random.choice(indArr,size=int(minOcc),replace=True)
			keepArr = keepArr.tolist()
			indKeep=indKeep+keepArr
		else:
			indKeep=indKeep+catInd[str(cat)]

	balData = df.loc[indKeep,:].copy(deep=True)
	return balData

def StandardizeData(df,colStandard):
	'''
	The purpose of this function is to standardize select columns of a dataframe
	There are two inputs
		df: pandas dataframe

		colStandard: These are the names of the columns of the pandas dataframe 
		that should be standardized
	'''
	dfStand = df.copy()
	for entry in colStandard:
		stdC = np.std(df[entry])
		meanC = np.mean(df[entry])
		dfStand[entry]=(dfStand[entry]-meanC)/(stdC)

	return dfStand

def CreateROC(probs,col,lab,ax_array,yTrue):
	'''
	The purpose of this function is to compare the 
	'''
	thresh = np.linspace(0,1,100)
	numClass = probs.shape[1]

	for iC in np.arange(0,numClass):
		# Used to store the True/False Positive Rate
		TPR = np.zeros(thresh.shape)
		FPR = np.zeros(thresh.shape)

		# Used to calculate TPR/FPR
		yROC = np.zeros(yTrue.shape)
		yROC[np.argwhere(yTrue==iC)]=1

		for it,T in enumerate(thresh):
			yPred = probs[:,iC]>=T

			# Calculate FP, TP, FN, TN
			fp = np.sum((yPred == 1) & (yROC == 0))
			tp = np.sum((yPred == 1) & (yROC == 1))

			fn = np.sum((yPred == 0) & (yROC == 1))
			tn = np.sum((yPred == 0) & (yROC == 0))

			TPR[it] = tp/(tp+fn)
			FPR[it] = fp/(fp+tn)

		ax_array[iC].plot(FPR,TPR,color=col,label=lab)
		ax_array[iC].set_title("Class"+str(iC))
		ax_array[iC].set_xlabel("FPR")
		ax_array[iC].set_ylabel("TPR")
		ax_array[iC].legend(loc=4,prop={'size':10},markerscale=3)
		ax_array[iC].set_xticks(np.arange(0,1,0.05),minor=True)
		ax_array[iC].set_yticks(np.arange(0,1,0.05),minor=True)
		ax_array[iC].grid(visible=True,which='both',axis='both')
			











