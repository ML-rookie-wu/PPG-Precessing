# -*- coding: utf-8 -*-

"""
@author: Mark Wu
@file: PeakCompare.py
@time: 2023/3/15 18:59
"""


import os
import matplotlib.pyplot as plt
import numpy as np
import scipy
from codes.PaperTests.FrequenceAnalysis import new_peak_detect
from codes.PaperTests.BRApneaAnalysis import peak_detect, get_csv_data
from codes.utils.MyFilters import bandpass_filter

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号



def new_get_peak(filtered_data):
    # butter_start = bandpass_filter(data, fs=100, start_fs=0.1, end_fs=4)
    interval = 100 * 0.5
    start_peak_list1 = new_peak_detect(filtered_data, period_num=1.5, interval=interval)
    start_peak_list2 = new_peak_detect(filtered_data, period_num=1.03, interval=interval)
    print(start_peak_list1)
    print(start_peak_list2)
    start_peak_list = sorted(list(set(start_peak_list1 + start_peak_list2)))
    return start_peak_list


def threshold(data):
    peaks, _ = scipy.signal.find_peaks(data, height=(np.max(data) - np.min(data)) * 0.2)
    return peaks


def test1(dir_path):
    for root_path, dir_name, filenames in os.walk(dir_path):
        for filename in filenames:
            file_path = os.path.join(root_path, filename)
            print(file_path)
            data = get_csv_data(file_path)
            ir = data.ir
            buttered_data = bandpass_filter(ir, fs=100, start_fs=0.1, end_fs=4)
            peak_list1 = new_get_peak(buttered_data)
            value1 = [buttered_data[x] for x in peak_list1]
            peak_list2 = peak_detect(buttered_data, interval=50)
            value2 = [buttered_data[y] for y in peak_list2]
            peak_list3 = threshold(buttered_data)
            value3 = [buttered_data[z] for z in peak_list3]

            plt.figure(figsize=(10, 8))
            plt.subplot(211)
            plt.plot(buttered_data)
            plt.scatter(peak_list1, value1, color="red")
            plt.title("改进差分阈值法")
            plt.subplot(212)
            plt.plot(buttered_data)
            plt.scatter(peak_list2, value2, color="cyan")
            plt.title("动态差分阈值法")
            # plt.subplot(313)
            # plt.plot(buttered_data)
            # plt.scatter(peak_list3, value3, color="purple")
            # plt.title("threshold method")
            # plt.subplots_adjust(hspace=0.5)
            plt.show()


if __name__ == '__main__':

    dir_path = r"D:\my_projects_V1\my_projects\PPG_V1\data\BR\LSTMDataset\all_data"
    test1(dir_path)
