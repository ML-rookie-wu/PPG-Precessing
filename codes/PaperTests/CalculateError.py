# -*- coding: utf-8 -*-

"""
@author: Mark Wu
@file: CalculateError.py
@time: 2023/2/13 10:07
"""
import pandas as pd
import openpyxl
import numpy as np
from codes.utils.SaveToExcel import add_sheet_to_excel


def cal_rmse(data):
    rmse = np.sqrt(sum([x**2 for x in data]) / len(data))
    return rmse

def cal_mse(data):
    mse = sum([x**2 for x in data]) / len(data)
    return mse

def cal_abs_mean(data):
    abs_mean = np.mean([abs(x) for x in data])
    return abs_mean

def read_excel(path):
    data = pd.read_excel(path, engine="openpyxl")
    print(data.columns)
    return data

def group(data):
    my_group = {}
    tang_group = {}
    for i in range(len(data)):
        # print(data.iloc[i], type(data.iloc[i]))
        cur_spo2 = data.iloc[i].RealSpo2
        cur_myerror = data.iloc[i].MyError
        cur_tangerror = data.iloc[i].TangError
        if my_group.get(cur_spo2) is None:
            my_group[cur_spo2] = [cur_myerror]
        else:
            my_group[cur_spo2].append(cur_myerror)
        if tang_group.get(cur_spo2) is None:
            tang_group[cur_spo2] = [cur_tangerror]
        else:
            tang_group[cur_spo2].append(cur_tangerror)

    return my_group, tang_group

def diff_model_group(data):
    first_group = {}
    second_group = {}
    third_group = {}

    for i in range(len(data)):
        # print(data.iloc[i], type(data.iloc[i]))
        cur_spo2 = data.iloc[i].RealSpo2

        first_error = data.iloc[i].FirstError
        second_error = data.iloc[i].SecondError
        third_error = data.iloc[i].ThirdError

        if first_group.get(cur_spo2) is None:
            first_group[cur_spo2] = [first_error]
        else:
            first_group[cur_spo2].append(first_error)

        if second_group.get(cur_spo2) is None:
            second_group[cur_spo2] = [second_error]
        else:
            second_group[cur_spo2].append(second_error)

        if third_group.get(cur_spo2) is None:
            third_group[cur_spo2] = [third_error]
        else:
            third_group[cur_spo2].append(third_error)

    return first_group, second_group, third_group

def main():
    path = r'D:\my_projects_V1\my_projects\PPG_V1\results\paper_test\5disturb_compare.xlsx'
    data = read_excel(path)
    my_error_res = []
    tang_error_res = []
    my_error = data.MyError
    tang_error = data.TangError
    all_my_rmse = cal_rmse(my_error)
    all_tang_rmse = cal_rmse(tang_error)
    all_my_absmean = cal_abs_mean(my_error)
    all_tang_absmean = cal_abs_mean(tang_error)
    print(all_my_rmse, all_tang_rmse)
    print(all_my_absmean, all_tang_absmean)

    my_group, tang_group = group(data)
    for key, value in my_group.items():
        # print(key, cal_abs_mean(value), cal_rmse(value))
        abs_mean = cal_abs_mean(value)
        rmse = cal_rmse(value)
        my_error_res.append([key, abs_mean, rmse])
    print("tang error")
    for key, value in tang_group.items():
        # print(key, cal_abs_mean(value), cal_rmse(value))
        abs_mean = cal_abs_mean(value)
        rmse = cal_rmse(value)
        tang_error_res.append([key, abs_mean, rmse])
    my_df = pd.DataFrame(my_error_res)
    my_df.columns = ["spo2", "abs_mean", "rmse"]
    tang_df = pd.DataFrame(tang_error_res)
    tang_df.columns = ["spo2", "abs_mean", "rmse"]
    excel_path = r'D:\my_projects_V1\my_projects\PPG_V1\results\paper_test\5disturb_compare.xlsx'
    sheet_name1 = "my_error"
    sheet_name2 = "tang_error"
    add_sheet_to_excel(my_df, excel_path, sheet_name1)
    add_sheet_to_excel(tang_df, excel_path, sheet_name2)

def diff_model():
    path = r'D:\my_projects_V1\my_projects\PPG_V1\results\paper_test\model_compare\5disturb_model.xlsx'
    data = read_excel(path)
    error_res = []

    first_error = data.FirstError
    second_error = data.SecondError
    third_error = data.ThirdError

    all_first_rmse = cal_rmse(first_error)
    all_first_absmean = cal_abs_mean(first_error)

    all_second_rmse = cal_rmse(second_error)
    all_second_absmean = cal_abs_mean(second_error)

    all_third_rmse = cal_rmse(third_error)
    all_third_absmean = cal_abs_mean(third_error)

    print(all_first_absmean, all_second_absmean, all_third_absmean)
    print(all_first_rmse, all_second_rmse, all_third_rmse)

    first_group, second_group, third_group = diff_model_group(data)
    print(third_group)
    keys = first_group.keys()
    for key in keys:
        first_absmean = cal_abs_mean(first_group[key])
        second_absmean = cal_abs_mean(second_group[key])
        third_absmean = cal_abs_mean(third_group[key])

        first_rmse = cal_rmse(first_group[key])
        second_rmse = cal_rmse(second_group[key])
        third_rmse = cal_rmse(third_group[key])

        error_res.append([key, first_absmean, second_absmean, third_absmean, first_rmse, second_rmse, third_rmse])

    my_df = pd.DataFrame(error_res)
    my_df.columns = ["spo2", "first_absmean", "second_absmean", "third_absmean", "first_rmse", "second_rmse", "third_rmse"]

    excel_path = r'D:\my_projects_V1\my_projects\PPG_V1\results\paper_test\model_compare\5disturb_model.xlsx'
    sheet_name1 = "diff_model_error"

    add_sheet_to_excel(my_df, excel_path, sheet_name1)




if __name__ == '__main__':
    # main()
    diff_model()