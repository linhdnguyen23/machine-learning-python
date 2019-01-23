# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1]
Y = dataset.iloc[:, 3]

# Taking care of missing data by filling it with the average
from sklearn.impute import SimpleImputer
import sklearn
 
imputer = SimpleImputer (missing_values = np.nan, strategy = 'mean')
 
imputer = imputer.fit(X.iloc[:, 1:3])
print('The scikit-learn version is {}.'.format(sklearn.__version__))
 
X.iloc[:, 1:3] = imputer.transform(X.iloc[:, 1:3])
# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
label_enc_X = LabelEncoder()
X.iloc[:, 0] = label_enc_X.fit_transform(X.values[:, 0])
# categorical feature is deprecated, use column transformer
# ohe = OneHotEncoder(categorical_features = [0])
ohe = OneHotEncoder()
col_transformer = ColumnTransformer([("one hot encoder", OneHotEncoder(), [0])], remainder = 'passthrough')
X = col_transformer.fit_transform(X)
# categorical feature is deprecated, use column transformer
# X = ohe.fit_transform(X).toarray()
label_enc_Y = LabelEncoder()
Y = label_enc_Y.fit_transform(Y)
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 2, random_state = 0)

# Feature scaling
from sklearn.preprocessing import StandardScaler
scale_X = StandardScaler()
X_train = scale_X.fit_transform(X_train)
X_test = scale_X.fit_transform(X_test)

