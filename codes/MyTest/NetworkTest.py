# -*- coding: utf-8 -*-

"""
@author: Mark Wu
@file: NetworkTest.py
@time: 2023/2/11 11:49
"""

from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import pandas as pd
import openpyxl
import numpy as np


path = r'D:\my_projects_V1\my_projects\PPG_V1\results\errors\errors.xlsx'
data = pd.read_excel(path, engine="openpyxl")
print(data.columns)
ss = StandardScaler()
# data = ss.fit_transform(data)
# print(data)
x = data.iloc[:, 1:]
y = data.iloc[:, 1]
# print(x)
# print(y)
train_data, test_data, train_target, test_target = train_test_split(x, y, test_size=0.3, random_state=42)
std = StandardScaler()
train_data_std = std.fit_transform(train_data)
test_data_std = std.transform(test_data)
# regr = MLPRegressor(solver='sgd', hidden_layer_sizes=(10, 10), activation='identity', max_iter=500).fit(train_data_std, train_target)   # 结果一般
regr = MLPRegressor(solver='sgd', hidden_layer_sizes=(3, 10), activation='identity', max_iter=500).fit(train_data_std, train_target)

# print(test_data)
predic_train = regr.predict(train_data_std)
y_pred = regr.predict(test_data_std)
print(test_target)
print(y_pred)
mse1 = mean_squared_error(predic_train, train_target)
mse2 = mean_squared_error(y_pred, test_target)
print(mse1, mse2)
print(mse2/len(test_data))