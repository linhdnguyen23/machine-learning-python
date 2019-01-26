# Data Preprocessing Template

# Importing the libraries
import numpy as np 
import pandas as pd
import matplotlib as plt

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values
