# -*- coding: utf-8 -*-

"""
@author: Mark Wu
@file: ApneaTest.py
@time: 2023/2/23 9:24
"""


import matplotlib.pyplot as plt
import numpy as np
from codes.utils.CaculateSpo2 import cal_real_spo2, cal_spo2
from codes.utils.GetFileData import travel_dir, read_from_file
from codes.utils.MyFilters import dwt, bandpass_filter
from codes.utils.GetRvalue import get_R



def peak_detect(filtered_data, interval=200):
    peakList = []
    ir2_max = np.max(filtered_data)
    ir2_min = np.min(filtered_data)
    ir2_mean = (sum(filtered_data) - ir2_max - ir2_min) / (len(filtered_data) - 2)

    diff = np.diff(filtered_data)
    diff_max = np.max(diff)
    diff_min = np.min(diff)
    diff_mean = (sum(diff) - diff_max - diff_min) / (len(diff) - 2)

    # th1 = 1.2 * diff_mean
    # normal_ir2 = normalization(filtered_data)
    # normal_diff = normalization(diff)

    for i in range(len(diff) - 2):
        if diff[i] > 0 and diff[i + 1] < 0:
            index = i + 1
            for j in range(0, 4):
                if index+j < len(filtered_data) and filtered_data[index + j] - filtered_data[index + j - 1] > 0 and filtered_data[index + j + 1] - filtered_data[index + j] < 0:
                    peak_index = index + j
                    # print(peak_index)
                    if len(peakList) == 0:
                        peakList.append(peak_index)
                    else:
                        if peak_index - peakList[-1] < interval:
                            break
                        else:
                            peakList.append(peak_index)
    return peakList

def cal_hr(peakList, frequency=500):
    ppTime = np.diff(peakList)
    intervel_max = max(ppTime)
    intervel_min = min(ppTime)
    # print(ppTime)
    hr = frequency / ((np.sum(ppTime) - intervel_max - intervel_min) / (len(ppTime) - 2)) * 60
    return round(hr, 0)

def cal_one_data(file_path):
    data = read_from_file(file_path)
    res = []
    hr_res = []
    ir2 = data.ir2
    red2 = data.red2
    start = 0
    step = 500
    window = 4000
    end = start + window
    while end <= len(ir2):
        # 小波
        dwted_ir2 = dwt(ir2[start: end])
        dwted_red2 = dwt(red2[start: end])
        R = get_R(dwted_ir2, dwted_red2, ir2, red2)
        spo2_third, spo2_second, spo2 = cal_spo2(R)
        res.append(spo2_third)

        # 心率
        filtered_temp_data = bandpass_filter(ir2[start:end], start_fs=0.5, end_fs=3)
        # plt.plot(filtered_temp_data)
        # plt.show()
        peaks_list = peak_detect(filtered_temp_data, interval=200)
        # print(peaks_list)
        hr = cal_hr(peaks_list)
        hr_res.append(hr)
        start += step
        end = start + window

    plt.figure(figsize=(10, 8))
    plt.subplot(411)
    plt.plot(ir2)
    plt.subplot(412)
    plt.plot(data.resp)
    plt.subplot(413)
    plt.plot(res)
    plt.subplot(414)
    plt.plot(hr_res)
    plt.show()
    print(res)

def main():
    # path = r"D:\my_projects_V1\my_projects\PPG_V1\data\real_data\wu\apnea\20230223090433.txt"
    path = r'D:\my_projects_V1\my_projects\PPG_V1\data\real_data\wu\old\20230225201425.txt'
    # path = r'D:\my_projects_V1\my_projects\PPG_V1\data\spo2_compare_test\0disturb_3pulse\99\20221130184012_99.txt'
    cal_one_data(path)


if __name__ == '__main__':
    main()

