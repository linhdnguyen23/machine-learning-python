# Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

# Taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
label_enc_X = LabelEncoder()
X.iloc[:, 3] = label_enc_X.fit_transform(X.iloc[:, 3])
# categorical feature is deprecated, use column transformer
# ohe = OneHotEncoder(categorical_features = [0])
ohe = OneHotEncoder()
col_transformer = ColumnTransformer([("one hot encoder", OneHotEncoder(), [3])], remainder = 'passthrough')
X = col_transformer.fit_transform(X)
# categorical feature is deprecated, use column transformer
# X = ohe.fit_transform(X).toarray()
label_enc_Y = LabelEncoder()
Y = label_enc_Y.fit_transform(Y)