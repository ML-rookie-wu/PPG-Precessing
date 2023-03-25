# -*- coding: utf-8 -*-

"""
@author: Mark Wu
@file: FeatureSelect.py
@time: 2023/3/17 11:35
"""
import pandas as pd
import numpy as np
from codes.PaperTests.Model_Regression import my_logistic
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, ShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
# from sklearn.model_selection import cross_val_score, ShuffleSplit
from sklearn.ensemble import RandomForestRegressor


def model_test():
    path = r"D:\my_projects_V1\my_projects\PPG_V1\data\BR\LSTMDataset\features.csv"
    data = pd.read_csv(path, sep=",")
    # model = svm.SVC(kernel="linear", decision_function_shape="ovo")
    # model = RandomForestClassifier()
    # model = KNeighborsClassifier()
    model = LogisticRegression()
    y = data.label
    x = data.iloc[:, 1:]
    train_data, test_data, train_target, test_target = train_test_split(x, y, test_size=0.3, random_state=42)
    std = StandardScaler()
    train_data_std = std.fit_transform(train_data)
    test_data_std = std.transform(test_data)

    lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(train_data_std, train_target)
    model = SelectFromModel(lsvc, prefit=True)
    X_new = model.transform(train_data_std)
    print(X_new.shape)

    # print(train_data_std.shape, train_target.shape, test_data_std.shape, test_target.shape)
    # my_svm(train_data_std, train_target, test_data_std, test_target)
    # my_logistic(train_data_std, train_target, test_data_std, test_target)
    # my_random_forest(train_data_std, train_target, test_data_std, test_target)
    # my_knn(train_data_std, train_target, test_data_std, test_target)
    # my_bayes(train_data_std, train_target, test_data_std, test_target)

    # model.fit(train_data_std, train_target)
    # # score = model.score(test_data_std, test_target)
    # acu_train = model.score(train_data_std, train_target)
    # acu_test = model.score(test_data_std, test_target)
    # y_pred = model.predict(test_data_std)
    # recall = recall_score(test_target, y_pred, average="macro")
    # print(acu_train, acu_test, recall)

    rf = RandomForestRegressor()

    scores = []
    names = x.columns
    # for i in range(train_data_std.shape[1]):
    #     score = cross_val_score(rf, train_data_std[:, i:i + 1], train_target, scoring="r2",
    #                             cv=ShuffleSplit(len(train_data_std), test_size=.3, random_state=3))
    #     scores.append((round(np.mean(score), 3), names[i]))
    # print(sorted(scores, reverse=True))
    # rf.fit(train_data_std, train_target)
    print(sorted(zip(map(lambda x: round(x, 4), rf.feature_importances_), names),
                 reverse=True))
    print("Scores for X0, X1, X2:", list(map(lambda x: round(x, 3),
                                        rf.feature_importances_)))


if __name__ == '__main__':
    model_test()