# Multiple Linear Regression
# Importing the numpy library
import numpy as np
# Importing the matlab library
import matplotlib.pyplot as plt
# Importing the pandas library
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, : -1]
y = dataset.iloc[:, -1].values
# Taking care of missing data by filling it with the average

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
label_enc_X = LabelEncoder()
X.iloc[:, 3] = pd.to_numeric(label_enc_X.fit_transform(X.iloc[:, 3]))
# categorical feature is deprecated, use column transformer
# ohe = OneHotEncoder(categorical_features = [0])
ohe = OneHotEncoder()
col_transformer = ColumnTransformer([("one hot encoder", OneHotEncoder(), [3])], remainder = 'passthrough')
X = col_transformer.fit_transform(X)

# Avoid dummy variable
X = X[:, 1:]

# Split data into train and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Fit multiple linear regression to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_prediction = regressor.predict(X_test)
 
import statsmodels.formula.api as sm
X = np.append(arr = np.ones((50, 1)).astype(int), values = X, axis = 1)
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 1, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()


X_opt = X[:, [0, 3, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 3]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()