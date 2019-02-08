#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 19:48:41 2019

@author: linh-nguyen
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model

df = pd.read_csv("/home/linh-nguyen/Documents/Coursera/FuelConsumption.csv")
cdf = df[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB', 'CO2EMISSIONS',
          'FUELCONSUMPTION_CITY', 'FUELCONSUMPTION_HWY']]

plt.close()

mask = np.random.rand(len(df)) < 0.8
train = cdf[mask]
test = cdf[~mask]

regr = linear_model.LinearRegression()

x = np.asanyarray(train[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB']])
y = np.asanyarray(train[['CO2EMISSIONS']])
regr.fit(x, y)

regr1 = linear_model.LinearRegression()
x1 = np.asanyarray(train[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_CITY']])
y1 = y
regr1.fit(x1, y1)

regr2 = linear_model.LinearRegression()
x2 = np.asanyarray(train[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_HWY']])
y2 = y
regr2.fit(x2, y2)

# The coefficients
print('Coefficients: ', regr.coef_)

x = np.asanyarray(test[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB']])
y = np.asanyarray(test[['CO2EMISSIONS']])
y_hat = regr.predict(x)
print("Residual sum of squares: %.2f" % np.mean((y_hat - y) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % regr.score(x, y))

x1 = np.asanyarray(test[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_CITY']])
y1 = np.asanyarray(test[['CO2EMISSIONS']])
y_hat1 = regr1.predict(x1)
print("Residual sum of squares using FUELCONSUMPTION_CITY: %.2f" % np.mean((y_hat1 - y) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % regr1.score(x1, y1))

x2 = np.asanyarray(test[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_HWY']])
y2 = np.asanyarray(test[['CO2EMISSIONS']])
y_hat2 = regr2.predict(x2)
print("Residual sum of squares using FUELCONSUMPTION_HWY: %.2f" % np.mean((y_hat2 - y) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % regr2.score(x2, y2))