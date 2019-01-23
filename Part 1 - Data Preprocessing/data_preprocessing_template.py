# Importing the numpy library
import numpy as np

# Importing the matlab library
import matplotlib.pyplot as plt

# Importing the pandas library
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1]
Y = dataset.iloc[:, 3]

# Taking care of missing data by filling it with the average

# Encoding categorical data

# Split data into train and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 2, random_state = 0)

# Feature scaling
from sklearn.preprocessing import StandardScaler
scale_X = StandardScaler()
X_train = scale_X.fit_transform(X_train)
X_test = scale_X.fit_transform(X_test)

