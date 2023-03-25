# -*- coding: utf-8 -*-

"""
@author: Mark Wu
@file: Model.py
@time: 2023/3/7 9:36
"""

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR  # SVM中的回归算法
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import BaggingRegressor



def svr(x, y):
    model = SVR()
    # y = np.array(y).reshape(-1, 1)
    train_data, test_data, train_target, test_target = train_test_split(x, y, test_size=0.3, random_state=42)
    std = StandardScaler()
    train_data_std = std.fit_transform(train_data)
    test_data_std = std.transform(test_data)
    model.fit(train_data_std, train_target)
    score = model.score(test_data_std, test_target)
    print(score)

def decision_tree(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
    std = StandardScaler()
    train_data_std = std.fit_transform(x_train)
    test_data_std = std.transform(x_test)
    clf = DecisionTreeRegressor()
    rf = clf.fit(train_data_std, y_train.ravel())
    y_pred = rf.predict(test_data_std)
    print("DecisionTreeRegressor结果如下：")
    print("训练集分数：", rf.score(train_data_std, y_train))
    print("验证集分数：", rf.score(test_data_std, y_test))

def random_forest(x, y):

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    std = StandardScaler()
    train_data_std = std.fit_transform(x_train)
    test_data_std = std.transform(x_test)
    clf = RandomForestRegressor(n_estimators=500)
    rf = clf.fit(x_train, y_train.ravel())
    y_pred = rf.predict(x_test)
    print("RandomForestRegressor结果如下：")
    print("训练集分数：", rf.score(x_train, y_train))
    print("验证集分数：", rf.score(x_test, y_test))

def adboost(x, y):

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    clf = AdaBoostRegressor()
    rf = clf.fit(x_train, y_train.ravel())
    y_pred = rf.predict(x_test)
    print("AdaBoostRegressor结果如下：")
    print("训练集分数：", rf.score(x_train, y_train))
    print("验证集分数：", rf.score(x_test, y_test))

def bagging(x, y):

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    clf = BaggingRegressor()
    rf = clf.fit(x_train, y_train.ravel())
    y_pred = rf.predict(x_test)
    print("BaggingRegressor结果如下：")
    print("训练集分数：", rf.score(x_train, y_train))
    print("验证集分数：", rf.score(x_test, y_test))


def mlpregressor(x, y):

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    clf = MLPRegressor(solver='lbfgs', hidden_layer_sizes=(6, 3), activation='logistic', max_iter=500)
    rf = clf.fit(x_train, y_train.ravel())
    y_pred = rf.predict(x_test)
    print("MLPRegressor结果如下：")
    print("训练集分数：", rf.score(x_train, y_train))
    print("验证集分数：", rf.score(x_test, y_test))
