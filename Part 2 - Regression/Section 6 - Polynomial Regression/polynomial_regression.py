# Data Preprocessing Template

# Importing the libraries
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Fitting linear regression to the dataset
from sklearn.linear_model import LinearRegression
linear_reg = LinearRegression()
linear_reg.fit(X, y)

# Fitting polynomial regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
polynomial_reg = PolynomialFeatures(degree = 4)
X_poly = polynomial_reg.fit_transform(X)
linear_reg_2 = LinearRegression()
linear_reg_2.fit(X_poly, y)

# Visualizing the linear regression model
y_pred_linear = linear_reg.predict(X)
plt.scatter(X, y, color = 'red')
plt.plot(X, y_pred_linear, color = 'blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()


# Visualizing the polynomial regression model
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = np.reshape(X_grid, (len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, linear_reg_2.predict(polynomial_reg.fit_transform(X_grid)), color = 'blue')
plt.title('Truth or Bluff (Poly Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

linear_reg.predict(np.array(6.5).reshape(1, -1))

linear_reg_2.predict(polynomial_reg.fit_transform(np.array(6.5).reshape(1, -1)))