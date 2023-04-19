#coding:utf-8

from codes.utils.MyFilters import MyProcess
from codes.utils.GetFileData import read_from_file, travel_dir
from codes.utils.GetRvalue import get_R
import os
import openpyxl
import pandas as pd
import random


def spo2_validate(value):
    value = round(value)
    if value > 100:
        value = 100
    return value


def cal_spo2(R):
    # 三元回归
    # spo2_third = -115.46 + 735.1 * R - 809.24 * (R ** 2) + 281.71 * (R ** 3)

    # spo2_third = 101.26 + 29.5 * R - 100.56 * (R ** 2) + 36.59 * (R ** 3)      # 3disturb_5pulse_test

    # spo2_third = 93.29 + 58.64 * R - 118.46 * (R ** 2) + 47.62 * (R ** 3)    # all_results

    spo2_third = 108.56925 - 10.38594 * R - 18.63495 * (R ** 2) - 0.46472 * (R ** 3)
    spo2_third = spo2_validate(spo2_third)

    # 二元回归
    # spo2_second = 99.39 + fast.13 * R - 26.91 * (R ** 2)

    # spo2_second = 111.2 - 17.54 * R - 13.99 * (R ** 2)         # 3disturb_5pulse_test

    # spo2_second = 110.59 - 16.63 * R - 13.07 * (R ** 2)         # all_results

    spo2_second = 108.41496 - 9.69548 * R - 18.63495 * (R ** 2)    # all results new
    spo2_second = spo2_validate(spo2_second)

    # 一元回归
    # spo2 = 122.69 - 30.45 * R

    # spo2 = 117.15 - 36.08 * R         # 3disturb_5pulse_test

    # spo2 = 117.25 - 35.74 * R         # all_results

    spo2 = 117.98444 - 37.7864 * R
    spo2 = spo2_validate(spo2)

    return spo2_third, spo2_second, spo2


def get_weight_spo2(spo2, spo2_second, spo2_third):
    pass


def cal_real_spo2(parent_dir):
    all_files = travel_dir(parent_dir)
    filter_func = MyProcess('dwt')
    res = []
    for file_path in all_files:
        spo2_value = int(os.path.split(file_path)[1].split(".")[0].split("_")[1])
        ir2, red2 = read_from_file(file_path)
        start = 1000
        step = 500
        window = 4000
        end = start + step
        while end <= len(ir2):
            # vmd
            # vmded_ir2, ir2_disturb = self.preprocess(ir2[start: end])
            # vmded_red2, red2_disturb = self.preprocess(red2[start: end])
            # R = self.get_R(vmded_ir2, vmded_red2, ir2_disturb, red2_disturb)

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
        excel_path = r'E:\my_projects\PPG\results\errors\errors_all.xlsx'
    wb = openpyxl.load_workbook(excel_path)
    writer = pd.ExcelWriter(excel_path, engine="openpyxl")
    writer.book = wb
    if sheet_name is None:
        sheet_name = "test_%s" % str(random.randint(1, 100))
    df.to_excel(writer, sheet_name=sheet_name, index=False)
    writer.save()


def main():
    parent_dir = r'E:\my_projects\PPG\data\spo2_compare\3disturb_3pulse'
    excel_path = r'E:\my_projects\PPG\results\errors\spo2_compare.xlsx'
    sheet_name = os.path.split(parent_dir)[1]
    results = cal_real_spo2(parent_dir)
    df = pd.DataFrame(results)
    df.columns = ["real_spo2", "third", "second", "first"]
    print(df)
    add_sheet_to_excel(df, excel_path, sheet_name)


if __name__ == '__main__':
    main()




