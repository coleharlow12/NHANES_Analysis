# The purpose of this repository is to analyze the data found in the NHANES III study. 

The data in the NHANES III study is available through the ICPSR which is part of the Institute for Social Research at University of Michigan. The data is part of NACDA which is
the National Archive of Computerized Data on Aging. When downloading the dataset be sure to 
download both the .sas and the .txt file. The sas file contains information relevant to load the text data and was used heavily in order to create the file DataPreparation/LoadData.py. 
Once downloaded both the .txt and .sas file should be added to the DataPreparation directory

The data is split into four separate folders: DataPreparation, LogisticRegression, and NeuroNet
	## DataPreparation: The database should be downloaded to this directory. Once the data is 
					   downloaded the file LoadData.py loads the txt file and stores it into 
					   a pandas dataframe which is then saved as a pkl format. PrepData.py 
					   should then be run and will be used to separate each respondant into a
					   category based on their height. PrepData.py also removed empty entries,
					   balances the dataset, and standardizes the non-target variables

    ## LogisticRegression: Contains two files to study the ability of logisticRegression to 
   					   classify the respondants heights. In the file trnLogRegOVR.py a 
   					   one versus rest architecture is used for training. In the file 
   					   trnLogRegMultistatic a one vs one architecture is used for training

    ## NeuroNet: 	   In this folder a neural network is trained to classify the heights of
    				   the respondants in NHANES III. The neural network is built using pyTorch and contains its own README.md in order to explain the structure of the directory

Dependencies: See the requirements.txt for the full list of dependencies with versions 
