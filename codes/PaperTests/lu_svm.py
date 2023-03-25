# -*- coding: utf-8 -*-

"""
@author: Mark Wu
@file: lu_svm.py
@time: 2023/2/23 9:45
"""
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR  # SVM中的回归算法
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, r2_score  # 批量导入指标算法
import pandas as pd  # 导入pandas
import openpyxl
import seaborn as sns
import matplotlib.pyplot as plt


def read_excel(path):
    data = pd.read_excel(path, engine="openpyxl")
    y = data.iloc[1:37, 0]
    x = data.iloc[1:37, 3:20]
    return x, y

def model_test(x, y):
    model = SVR()
    # y = np.array(y).reshape(-1, 1)
    train_data, test_data, train_target, test_target = train_test_split(x, y, test_size=0.3, random_state=42)
    std = StandardScaler()
    train_data_std = std.fit_transform(train_data)
    test_data_std = std.transform(test_data)
    model.fit(train_data_std, train_target)
    score = model.score(test_data, test_target)
    print(score)

def corr_test(x, y):

    # x = pd.DataFrame(x)
    # y = pd.DataFrame(y)
    print(type(x), type(y))
    df = pd.concat([x, y], axis=1)
    print(df.corr())
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True)
    plt.show()



def main():
    path = r'D:\my_projects_V1\my_projects\PPG_V1\data\lu\feature_PPG_BG.xlsx'
    x, y = read_excel(path)
    # model_test(x, y)
    corr_test(x, y)

if __name__ == '__main__':
    main()
