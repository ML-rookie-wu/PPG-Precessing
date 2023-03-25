# -*- coding: utf-86 -*-
# @Time : 2022/10/19 20:46
# @Author : Mark Wu
# @FileName: all_test.py
# @Software: PyCharm
import json
import math
import os
import random
import pandas as pd
from pathlib import Path
import numpy as np
import scipy.signal as scisignal
from codes.process import bandpass_filter
from codes.read_data import vmd
import matplotlib.pyplot as plt
import openpyxl


def read_data(path):
    data = pd.read_table(path, sep=',')
    data.columns = ["ir1", "red1", "ir2", "red2", "resp"]
    return data

def save_to_excel(self, file_name: str):
    # file_name = "R.xlsx"
    path = os.path.join(self.results_path, file_name)
    writer = pd.ExcelWriter(path, engine="openpyxl")
    results = self.deal()
    df = pd.DataFrame(results)
    df.to_excel(writer, index=False)
    writer.save()

def perfusion_test():
    save_path = ""
    writer = pd.ExcelWriter(r'E:\my_projects\PPG\results\perfusion.xlsx', engine='openpyxl')
    for i in range(6):
        dir_path = Path('D:\\works\\disturb\\{}disturb_pulse'.format(str(i)))
        print(dir_path)
        res = []
        for path in dir_path.glob("*.txt"):
            pulse = float(os.path.splitext(path)[0].split("_")[-1])
            ir2 = read_data(path)["ir2"][1000:3000]
            # filtered_ir2 = bandpass_filter(ir2)
            filtered_ir2 = vmd(ir2)
            y_max = np.max(filtered_ir2)
            y_min = np.min(filtered_ir2)
            x_mean = np.mean(ir2)
            perfusion = ((y_max - y_min) / x_mean) * 100
            # print("perfusion------------------", (pulse, round(perfusion, 1)))
            res.append([pulse, round(perfusion, 1)])
        df = pd.DataFrame(res)
        df.columns = ["pulse", "perfusion"]
        df.to_excel(writer, sheet_name="disturb{}".format(i), index=False)
    writer.save()

def read_excel():
    path = r'E:\my_projects\PPG\results\perfusion.xlsx'
    for i in range(6):
        data = pd.read_excel(path, sheet_name="disturb{}".format(str(i)), engine="openpyxl")
        x = list(range(len(data)))
        y1 = data["pulse"]
        y2 = data["perfusion"]
        plt.figure(figsize=(10, 8))
        plt.ylim(-1, 5)
        plt.scatter(x, y1, c="red", label="real")
        plt.scatter(x, y2, c="blue", label="calculate")
        plt.title("disturb{}".format(i))
        plt.legend()
        plt.show()

def read_R1():
    path = r'E:\my_projects\PPG\results\R.xlsx'
    data = pd.read_excel(path, engine="openpyxl")
    print('col----------', data.columns)
    x = data["spo2"]
    y = data["R"]
    plt.figure(figsize=(10, 8))
    plt.ylim(0, 2)
    plt.scatter(x, y, c="orange", label="R")
    plt.legend()
    plt.show()

def get_snr(data):
    x_mean = np.mean(data)
    # numerator = sum(x ** 2 for x in data)
    # denominator = sum((y - x_mean) ** 2 for y in data)
    return math.log((sum(x ** 2 for x in data) / sum((y - x_mean) ** 2 for y in data)), 10)

def get_rmse(raw_data, filtered_data):
    pass

def snr_test():
    for i in range(6):
        dir_path = Path('D:\\works\\disturb\\{}disturb_pulse'.format(str(i)))
        print(dir_path)
        res = []
        for path in dir_path.glob("*.txt"):
            pulse = float(os.path.splitext(path)[0].split("_")[-1])
            ir2 = read_data(path)["ir2"][1000:3000]
            snr = get_snr(ir2)
            filtered_data = vmd(ir2)
            filtered_snr = get_snr(filtered_data)
            print("snr = {}, filtered_snr = {}".format(snr, filtered_snr))

def add_sheet_to_excel(df, excel_path=None, sheet_name=None):

    if excel_path is None:
        excel_path = r'E:\my_projects\PPG\results\errors\errors_all.xlsx'
    wb = openpyxl.load_workbook(excel_path)
    writer = pd.ExcelWriter(excel_path, engine="openpyxl")
    writer.book = wb
    if sheet_name is None:
        sheet_name = "test_%s" % str(random.randint(1, 100))
    df.to_excel(writer, sheet_name=sheet_name, index=False)
    writer.save()

def error():

    path = r'E:\my_projects\PPG\results\cal_json\3disturb_3pulse.json'
    excel_path = r'E:\my_projects\PPG\results\errors\errors_all.xlsx'
    sheet_name = os.path.split(path)[1].split(".")[0]
    with open(path, 'r') as f:
        data = json.load(f)
    results = []
    for spo2, cal_values in data.items():
        real_spo2 = int(spo2)
        rmse_list = [real_spo2]
        for one in cal_values:
            rmse = math.sqrt(sum((x - real_spo2) ** 2 for x in one) / len(one))
            rmse_list.append(rmse)
        results.append(rmse_list)
    df = pd.DataFrame(results)
    df.columns = ["spo2_value", "one", "two", "three"]
    add_sheet_to_excel(df, excel_path, sheet_name)

def error_test():

    path = r'E:\my_projects\PPG\results\cal_json\1disturb_3pulse.json'
    excel_path = r'E:\my_projects\PPG\results\errors\errors.xlsx'
    sheet_name = os.path.split(path)[1].split(".")[0]
    with open(path, 'r') as f:
        data = json.load(f)
    results = []
    for spo2, cal_values in data.items():
        real_spo2 = int(spo2)
        # rmse_list = [real_spo2]
        # print(len(cal_values[0]), len(cal_values[1]), len(cal_values[2]))
        for i in range(len(cal_values[0])):
            # rmse = math.sqrt(sum((x - real_spo2) ** 2 for x in one) / len(one))
            cal_list = [real_spo2, round(cal_values[0][i]), round(cal_values[1][i]), round(cal_values[2][i])]
            results.append(cal_list)
        # results.append(rmse_list)
    df = pd.DataFrame(results)
    df.columns = ["spo2_value", "one", "two", "three"]

    add_sheet_to_excel(df, excel_path, sheet_name)


if __name__ == '__main__':
    # perfusion_test()

    # read_excel()

    # read_R1()

    # snr_test()

    # error()

    error_test()