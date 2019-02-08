#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 22:09:09 2019

@author: linh-nguyen
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("FuelConsumption.csv")
cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
plt.close()
plt.figure()
plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS, color = 'blue', edgecolor = 'black')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()

mask = np.random.rand(len(df)) < 0.8
train = cdf[mask]
test = cdf[~mask]

from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
train_x = np.asanyarray(train[['ENGINESIZE']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])

test_x = np.asanyarray(test[['ENGINESIZE']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])

poly = PolynomialFeatures(degree = 2)
train_x_poly = poly.fit_transform(train_x)

regr = linear_model.LinearRegression()
train_y_ = regr.fit(train_x_poly, train_y)

print("Coefficients: ", regr.coef_)
print("Intercept: ", regr.coef_)
XX = np.arange(0.0, 10.0, 0.1)
yy = regr.intercept_[0] + regr.coef_[0][1] * XX + regr.coef_[0][2] * np.power(XX, 2)
plt.plot(XX, yy, '-r' )

from sklearn.metrics import r2_score
test_x_poly = poly.fit_transform(test_x)
test_y_poly = regr.predict(test_x_poly)

print("Mean absolute error (degree = 2): %.2f" % np.mean(np.absolute(test_y - test_y_poly)))
print("Residual sum of squares (degree = 2): %.2f" % np.mean((test_y - test_y_poly) ** 2))
print("R2 score (degree = 2) is %.2f" % r2_score(test_y_poly, test_y))

poly1 = PolynomialFeatures(degree = 3)
train_x_poly1 = poly1.fit_transform(train_x)

regr1 = linear_model.LinearRegression()
regr1.fit(train_x_poly1, train_y)

test_x_poly1 = poly1.fit_transform(test_x)
test_y_poly1 = regr1.predict(test_x_poly1)

print("Coefficients: ", regr.coef_)
print("Intercept: ", regr.coef_)
yy1 = regr1.intercept_[0] + regr1.coef_[0][1] * XX + regr1.coef_[0][2] * np.power(XX, 2)
plt.plot(XX, yy1, '-g')

print("Mean absolute error (degree = 2): %.2f" % np.mean(np.absolute(test_y - test_y_poly1)))
print("Residual sum of squares (degree = 2): %.2f" % np.mean((test_y - test_y_poly1) ** 2))
print("R2 score (degree = 2) is %.2f" % r2_score(test_y_poly1, test_y))




