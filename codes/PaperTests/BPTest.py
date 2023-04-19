# -*- coding: utf-8 -*-

"""
@author: Mark Wu
@file: BPTest.py
@time: 2023/2/fast 13:53
"""

from codes.utils.MyFilters import MyProcess
from codes.utils.GetFileData import read_from_file, travel_dir
from codes.utils.GetRvalue import get_R
import os
import openpyxl
import pandas as pd
import random
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
import pickle
import numpy as np
import matplotlib.pyplot as plt


def spo2_validate(value):
    value = round(value)
    if value > 100:
        value = 100
    return value


def cal_spo2(R):
    # 三元回归
    spo2_third = 108.56925 - 10.38594 * R - 18.63495 * (R ** 2) - 0.46472 * (R ** 3)
    # spo2_third = spo2_validate(spo2_third)

    # 二元回归
    spo2_second = 108.41496 - 9.69548 * R - 18.63495 * (R ** 2)    # all results new
    # spo2_second = spo2_validate(spo2_second)

    # 一元回归
    spo2 = 117.98444 - 37.7864 * R
    # spo2 = spo2_validate(spo2)

    return spo2_third, spo2_second, spo2


def get_weight_spo2(spo2, spo2_second, spo2_third):
    pass


def cal_real_spo2(parent_dir):
    all_files = travel_dir(parent_dir)
    filter_func = MyProcess('dwt')
    res = []
    for file_path in all_files:
        print(file_path)
        spo2_value = int(os.path.split(file_path)[1].split(".")[0].split("_")[1])
        data = read_from_file(file_path)
        ir2 = data.ir2
        red2 = data.red2
        start = 500
        step = 1000
        window = 4000
        end = start + step
        while end <= len(ir2):
            # 小波
            dwted_ir2 = filter_func(ir2[start: end])
            dwted_red2 = filter_func(red2[start: end])
            R = get_R(dwted_ir2, dwted_red2, ir2, red2)
            spo2_third, spo2_second, spo2 = cal_spo2(R)
            res.append([spo2_value, spo2_third, spo2_second, spo2])
            start += step
            end = start + window
    return res

def add_sheet_to_excel(df, excel_path=None, sheet_name=None):

    if excel_path is None:
        excel_path = r'D:\my_projects_V1\my_projects\PPG_V1\results\paper_test\spo2_result'
    wb = openpyxl.load_workbook(excel_path)
    writer = pd.ExcelWriter(excel_path, engine="openpyxl")
    writer.book = wb
    if sheet_name is None:
        sheet_name = "test_%s" % str(random.randint(1, 100))
    df.to_excel(writer, sheet_name=sheet_name, index=False)
    writer.save()


def get_spo2_result():
    parent_dir = r'D:\my_projects_V1\my_projects\PPG_V1\data\paper_test\0disturb_3pulse'
    excel_path = r'D:\my_projects_V1\my_projects\PPG_V1\results\paper_test\spo2_result\spo2_compare.xlsx'
    sheet_name = os.path.split(parent_dir)[1]
    results = cal_real_spo2(parent_dir)
    df = pd.DataFrame(results)
    df.columns = ["real_spo2", "third", "second", "first"]
    # print(df)
    add_sheet_to_excel(df, excel_path, sheet_name)

def bp_model(excel_path):
    # path = r'D:\my_projects_V1\my_projects\PPG_V1\results\errors\errors.xlsx'
    data = pd.read_excel(excel_path, engine="openpyxl")
    print(data.columns)
    ss = StandardScaler()
    # data = ss.fit_transform(data)
    # print(data)
    x = data.iloc[:, 1:]    # data
    # print(x)
    y = data.iloc[:, 0]     # label
    # print(y)

    train_data, test_data, train_target, test_target = train_test_split(x, y, test_size=0.3, random_state=42)

    # print(train_target)
    std = StandardScaler()
    train_data_std = std.fit_transform(train_data)
    test_data_std = std.transform(test_data)
    # regr = MLPRegressor(solver='sgd', hidden_layer_sizes=(10, 10), activation='identity', max_iter=500).fit(train_data_std, train_target)   # 结果一般
    # regr = MLPRegressor(solver='adam', hidden_layer_sizes=(4,), activation='identity', max_iter=1000).fit(train_data_std, train_target)
    regr = MLPRegressor(solver='lbfgs', hidden_layer_sizes=(6, 3), activation='logistic', max_iter=1000).fit(train_data_std, train_target)

    # print(test_data)
    predic_train = regr.predict(train_data_std)
    y_pred = regr.predict(test_data_std)
    print(test_target)
    print(y_pred)
    mse1 = mean_squared_error(predic_train, train_target)
    mse2 = mean_squared_error(y_pred, test_target, squared=False)
    print(mse1, mse2)
    print(regr.score(test_data_std, test_target))
    loss = [test_target.to_list()[i] - list(y_pred)[i] for i in range(len(test_target))]
    error = np.sqrt(sum([x**2 for x in loss]) / len(loss))
    print("error", error, np.max(loss), np.min(loss))
    # plt.scatter(range(0, len(loss)), loss)
    # plt.show()

    # 保存模型
    # with open("./regr.pickle", "wb") as f:
    #     pickle.dump(regr, f)

def opit_test(excel_path):
    data = pd.read_excel(excel_path, engine="openpyxl")
    print(data.columns)
    ss = StandardScaler()
    x = data.iloc[:, 1:]  # data
    y = data.iloc[:, 0]  # label
    train_data, test_data, train_target, test_target = train_test_split(x, y, test_size=0.3, random_state=42)
    std = StandardScaler()
    train_data_std = std.fit_transform(train_data)
    test_data_std = std.transform(test_data)
    mlp_para = {"hidden_layer_sizes": [(6, 3)],
                "solver": ["lbfgs"],
                "max_iter": [1000],
                "activation": ["logistic"],
                "verbose": [True]
                }
    mlp = MLPRegressor()
    estimator = GridSearchCV(mlp, mlp_para, n_jobs=4)
    estimator.fit(train_data_std, train_target)
    print(estimator.best_score_)

def optimize_para(excel_path):
    data = pd.read_excel(excel_path, engine="openpyxl")
    print(data.columns)
    ss = StandardScaler()
    x = data.iloc[:, 1:]  # data
    y = data.iloc[:, 0]  # label
    train_data, test_data, train_target, test_target = train_test_split(x, y, test_size=0.3, random_state=42)
    std = StandardScaler()
    train_data_std = std.fit_transform(train_data)
    test_data_std = std.transform(test_data)
    mlp_para = {"hidden_layer_sizes": [(4,), (3, 2), (4, 3), (6, 3)],
                "solver": ["adam", "sgd", "lbfgs"],
                "max_iter": [20, 200, 400, 600, 800, 1000],
                "activation": ["relu", "logistic", "identity", "tanh"],
                "verbose": [True]
                }
    mlp = MLPRegressor()
    estimator = GridSearchCV(mlp, mlp_para, n_jobs=4)
    estimator.fit(train_data_std, train_target)
    # print(estimator.best_score_)

    # print(estimator.best_estimator_)

def optimized_compare(excel_path):
    # path = r'D:\my_projects_V1\my_projects\PPG_V1\results\errors\errors.xlsx'
    path = r"D:\my_projects_V1\my_projects\PPG_V1\results\paper_test\spo2_result\spo2_compare.xlsx"
    data = pd.read_excel(excel_path, engine="openpyxl")
    compare_data = pd.read_excel(path, engine="openpyxl", sheet_name="0disturb_3pulse")
    print(data.columns)

    x = data.iloc[:, 1:]  # data
    y = data.iloc[:, 0]  # label

    compare_x = compare_data.iloc[:, 1:]
    compare_y = compare_data.iloc[:, 0]

    train_data, test_data, train_target, test_target = train_test_split(x, y, test_size=0.3, random_state=42)

    std = StandardScaler()
    train_data_std = std.fit_transform(train_data)
    test_data_std = std.transform(test_data)

    compare_x_std = std.fit_transform(compare_x)
    regr = MLPRegressor(solver='lbfgs', hidden_layer_sizes=(6, 3), activation='logistic', max_iter=1000).fit(
        train_data_std, train_target)

    # print(test_data)
    predic_train = regr.predict(train_data_std)
    y_pred = regr.predict(test_data_std)

    compare_pred = regr.predict(compare_x_std)
    # print(test_target)
    # print(y_pred)
    mse1 = mean_squared_error(predic_train, train_target)
    mse2 = mean_squared_error(y_pred, test_target, squared=False)

    mse_compare = mean_squared_error(compare_pred, compare_y, squared=False)
    print(mse_compare)

    print(regr.score(test_data_std, test_target))
    loss = [test_target.to_list()[i] - list(y_pred)[i] for i in range(len(test_target))]
    error = np.sqrt(sum([x ** 2 for x in loss]) / len(loss))



def concat_all_sheet():
    excel_path = r'D:\my_projects_V1\my_projects\PPG_V1\results\paper_test\spo2_result\spo2_compare.xlsx'
    save_path = r'D:\my_projects_V1\my_projects\PPG_V1\results\paper_test\spo2_result\all_results.xlsx'
    sheets_name = pd.read_excel(excel_path, engine="openpyxl", sheet_name=None).keys()
    dfs = []
    for sheet_name in sheets_name:
        df = pd.read_excel(excel_path, engine="openpyxl", sheet_name=sheet_name)
        dfs.append(df)
    all_df = pd.concat(dfs, sort=False, ignore_index=True)
    all_df.to_excel(save_path, index=False)


def main():
    excel_path = r'D:\my_projects_V1\my_projects\PPG_V1\results\paper_test\spo2_result\all_results.xlsx'
    bp_model(excel_path)
    # optimize_para(excel_path)
    # opit_test(excel_path)
    # optimized_compare(excel_path)

if __name__ == '__main__':
    # get_spo2_result()
    # bp_model()
    # concat_all_sheet()
    main()


