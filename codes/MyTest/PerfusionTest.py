# -*- coding: utf-8 -*-

"""
@author: Mark Wu
@file: PerfusionTest.py
@time: 2023/2/5 11:13
"""
import os
import openpyxl
import math
import pandas as pd
import random
import time
import numpy as np
from codes.utils.GetFileData import travel_dir, read_from_file
from codes.utils.MyFilters import bandpass_filter, dwt

def add_sheet_to_excel(df, excel_path=None, sheet_name=None):

    if excel_path is None:
        curPath = os.path.dirname(__file__)
        rootPath = curPath[:curPath.find("PPG_V1")+len("PPG_V1\\")]
        resPath = os.path.join(rootPath, "results")
        if not os.path.exists(resPath):
            os.mkdir(os.path.join(rootPath, "results"))
        excel_name = time.strftime("%Y%m%d%H%M%S", time.localtime()) + ".xlsx"
        excel_path = os.path.join(resPath, excel_name)
    if not os.path.exists(excel_path):
        wb = openpyxl.Workbook()
        wb.save(excel_path)
    wb = openpyxl.load_workbook(excel_path)
    writer = pd.ExcelWriter(excel_path, engine="openpyxl")
    writer.book = wb
    if sheet_name is None:
        sheet_name = "test_%s" % str(random.randint(1, 100))
    df.to_excel(writer, sheet_name=sheet_name, index=False)
    writer.save()

def travel_data():
    parent_path = r'D:\my_projects_V1\my_projects\PPG_V1\data\pulse_compare\5disturb'
    all_files = travel_dir(parent_path)
    res = []
    for file_path in all_files:
        print(file_path)
        real_perfusion = os.path.split(file_path)[1].split("_")[1].split(".")[0]
        print(real_perfusion)
        data = read_from_file(file_path)
        ir2 = data.ir2
        red2 = data.red2
        start = 0
        window = 4000
        end = start + window
        step = 1000
        while end < len(ir2):
            win_data = ir2[start:end]
            ir2_mean = np.mean(win_data)
            ir2_max = max(win_data)
            ir2_min = min(win_data)
            perfusion = (ir2_max - ir2_min) / ir2_mean * 100
            print("perfusion = %s, mean = %s, max-min = %s" % (perfusion, ir2_mean, (ir2_max - ir2_min)))

            # filtered_wind = bandpass_filter(win_data, start_fs=0.1, end_fs=5)
            filtered_wind = dwt(win_data)
            ir_ac = math.sqrt(sum(x ** 2 for x in filtered_wind)) / window
            p = ir_ac / ir2_mean
            print("p =", p)

            res.append([perfusion, int(real_perfusion) / 10])
            start += step
            end = start + window

    # df = pd.DataFrame(res)
    # df.columns = ["cal_perfusion", "real_perfusion"]
    # path = r"D:\my_projects_V1\my_projects\PPG_V1\results\perfusion.xlsx"
    # add_sheet_to_excel(df, excel_path=path, sheet_name="5disturb")



def perfusion_test():
    file_path = r'D:\my_projects_V1\my_projects\PPG_V1\data\real_data\wu\20221205103726.txt'
    data = read_from_file(file_path)
    ir2 = data.ir2
    red2 = data.red2
    ir2_mean = np.mean(ir2)
    ir2_max = max(ir2)
    ir2_min = min(ir2)
    perfusion = (ir2_max - ir2_min) / ir2_mean * 100
    print("perfusion = %s, mean = %s, max-min = %s" % (perfusion, ir2_mean, (ir2_max - ir2_min)))


def main():
    travel_data()
    # perfusion_test()
    # curPath = os.path.dirname(__file__)
    # rootPath = curPath[:curPath.find("PPG_V1") + len("PPG_V1\\")]
    # print(rootPath)
    # print(time.strftime("%Y%m%d%H%M%S", time.localtime()))


if __name__ == '__main__':
    main()