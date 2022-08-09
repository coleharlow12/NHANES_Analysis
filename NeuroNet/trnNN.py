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
from utils import (
	CreateROC
	)

# Gets path to the files to be read
cwd = os.getcwd()
savePath = os.path.join(cwd,'PD_Data_Prepped.pkl')
data = pd.read_pickle(savePath)