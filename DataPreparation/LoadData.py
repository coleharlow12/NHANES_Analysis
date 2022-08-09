import pandas as pd
import numpy as np
import os	# Used for opening files and managing paths

# Gets path to the files to be read
cwd = os.getcwd()
sasPath = os.path.join(cwd,'02231-0002-Setup.sas')
dataPath = os.path.join(cwd,'02231-0002-Data.txt')
savePath = os.path.join(cwd,'PD_Data.pkl')

with open(sasPath) as f:
    lines = f.readlines()

# Finds the location of the INPUT and LABEL sections of the SAS file
iInp = 0
iLab = 0
for il,line in enumerate(lines):
	if "INPUT" in line:
		print("INPUT FOUND")
		iInp = il

	if "LABEL" in line:
		print("LABEL FOUND")
		iLab = il


labels = []
startI = []
endI = []
# Gets the indices and labels of all the data
for i0, iI in enumerate(np.arange(iInp+1,iLab-2)):
	rem = lines[iI].split()			#Gets the line and splits at whitespace
	labels.append(rem[0])			#First entry is the label
	numb = rem[1].split('-')		#Second entry gives datarange
	startI.append(int(numb[0]))		#Starting index
	endI.append(int(numb[-1]))		#Ending index
	#print(labels[i0],startI[i0],endI[i0])	#Used for data

#Specifies the labels I am interested in and finds their index
keep = ['BMPWT','BMPHT','BMPSITHT','BMPLEG','BMPARML','BMPBIAC','BMPBIIL']
keepInd = []
for item in keep:
	if item in labels:
		keepInd.append(labels.index(item))
	else:
		print("Item not found in list")

#Creates a empty pandas dataframe to load the data into
data = pd.DataFrame(columns=keep)

with open(dataPath) as d:
	lined = d.readlines()

i=0
for line in lined:
	addRow = {}
	for ind0,indK in enumerate(keepInd):
		#print(keep[ind0]) #DEBUG
		if startI[indK] is not endI[indK]:
			# Only appends data if it is all there
			try:
				addRow[keep[ind0]] = [float(line[(startI[indK]-1):(endI[indK])])]
			except ValueError:
				continue

		else:
			# Only appends data if it is all there
			try:
				addRow[keep[ind0]] = [float(line[startI[indK]])]
			except ValueError:
				continue

	addRow = pd.DataFrame(addRow,columns=keep)
	data = pd.concat([data,addRow],axis=0,ignore_index=True)
	print(i)
	i+=1

data.to_pickle(savePath)

