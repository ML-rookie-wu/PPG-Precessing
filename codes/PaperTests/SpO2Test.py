# -*- coding: utf-8 -*-

"""
@author: Mark Wu
@file: SpO2Test.py
@time: 2023/2/12 18:18
"""

import os

from codes.utils.GetFileData import travel_dir, read_from_file
from codes.utils.GetRvalue import get_R
from codes.utils.MyFilters import bandpass_filter, MyProcess
from codes.utils.SaveToExcel import save_to_excel

def spo2_validate(value):
    value = round(value)
    if value > 100:
        value = 100
    return value

def cal_spo2_compare(R):

    spo2_third = 108.56925 - 10.38594 * R - 18.63495 * (R ** 2) - 0.46472 * (R ** 3)
    # spo2_third = spo2_validate(spo2_third)

    # spo2_second = 108.41496 - 9.69548 * R - 18.63495 * (R ** 2)    # all results new
    # spo2_second = spo2_validate(spo2_second)
    #
    # spo2 = 117.98444 - 37.7864 * R
    # spo2 = spo2_validate(spo2)

    third_test = 11.78*(R**3) - 55.92*(R**2) + 28.84*R + 97.12

    return spo2_third, third_test

def cal_real_spo2(parent_dir):
    all_files = travel_dir(parent_dir)
    filter_func = MyProcess('dwt')
    res = []
    for file_path in all_files:
        print(file_path)
        spo2_value = int(os.path.split(file_path)[1].split(".")[0].split("_")[1])
        data = read_from_file(file_path)
        # ir2, red2 = read_from_file(file_path)
        ir2 = data.ir2
        red2 = data.red2
        start = 1000
        step = 1000
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
            # spo2_third, spo2_second, spo2 = cal_spo2(R)
            spo2_third, spo2_test = cal_spo2_compare(R)
            # res.append([spo2_value, spo2_third, spo2_second, spo2])
            # print(spo2_test, spo2_third)
            res.append([spo2_value, spo2_third, spo2_test])
            start += step
            end = start + window
    print(len(res))

    return res

def main():
    parent_dir = r'D:\my_projects_V1\my_projects\PPG_V1\data\paper_test\5disturb_3pulse'
    res = cal_real_spo2(parent_dir)
    save_path = r'D:\my_projects_V1\my_projects\PPG_V1\results\paper_test\5disturb_compare.xlsx'
    columns_name = ["RealSpo2", "MyModel", "TangModel"]
    save_to_excel(save_path, res, columns_name)


if __name__ == '__main__':
    main()

