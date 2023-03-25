# -*- coding: utf-8 -*-

"""
@author: Mark Wu
@file: HeartRate.py
@time: 2023/2/9 11:19
"""
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from codes.utils.MyFilters import bandpass_filter, reverse
from codes.utils.GetFileData import travel_dir
from codes.PaperTests.peakTest import read_data, peakTest
from codes.utils.SaveToExcel import save_to_excel
from codes.utils.GetFFT import signal_fft, get_freq
from codes.PaperTests.CalculateError import cal_abs_mean
from codes.PaperTests import PAPER_FIGURE_PATH

def get_files(dir_path):
    all_files = travel_dir(dir_path)
    return all_files

def hrTest(path, hr_real, frequency=500):
    ir2, red2 = read_data(path, label="real")
    start, end = 0, len(ir2)
    step = 500
    window = 15000
    hr_result = []
    while start < end:

        if start+window >= end:
            data = ir2[start: end]
        else:
            data = ir2[start: start+window]
        filtered_data = bandpass_filter(data, start_fs=0.1, end_fs=5)
        hr_data = bandpass_filter(data, start_fs=0.5, end_fs=3)
        f, absY = signal_fft(hr_data)
        freq, max_amp = get_freq(f, absY)

        # print(freq, freq*60)
        peakList = peakTest(filtered_data)
        peakValue = [filtered_data[i] for i in peakList]

        # plt.plot(filtered_data)
        # plt.scatter(peakList, peakValue, marker="o", color="red")
        # plt.show()

        # reversed_data = reverse(filtered_data)
        # valleyList = peakTest(reversed_data)

        p = len(peakList)
        hr1 = (p / 8) * 60

        ppTime = np.diff(peakList)
        intervel_max = max(ppTime)
        intervel_min = min(ppTime)
        # print(ppTime)
        hr = frequency / ((np.sum(ppTime) - intervel_max - intervel_min) / (len(ppTime)-2)) * 60
        freq_hr = freq * 60
        print("hr =", hr, hr_real, freq_hr)

        hr_result.append([hr_real, round(hr, 4), round(hr, 0), round(freq_hr, 0)])

        if start+window >= end:
            break
        start += step
        # end = start + window
    return hr_result

def main():
    dir_path = r'D:\my_projects_V1\my_projects\PPG_V1\data\hr_test\0disturb'
    all_file_path = get_files(dir_path)
    results = []
    for file_path in all_file_path:
        # print(file_path)
        hr_real = os.path.split(os.path.dirname(file_path))[1]
        file_hr_result = hrTest(file_path, hr_real)
        results += file_hr_result
    print(results)
    save_path = r"D:\my_projects_V1\my_projects\PPG_V1\results\paper_test\hr\hr_new_30.xlsx"
    column_name = ["real", "calculate", "int_value", "freq_cal"]
    save_to_excel(save_path, results, column_name)

def test():
    path = r"D:\my_projects_V1\my_projects\PPG_V1\data\hr_test\0disturb\110\20230209104101_110.txt"
    ir2, red2 = read_data(path)
    ir2 = bandpass_filter(ir2, start_fs=0.1, end_fs=5)
    plt.plot(ir2)
    plt.show()

def abs_mean():
    path = r"D:\my_projects_V1\my_projects\PPG_V1\results\paper_test\hr\hr_new_modify.xlsx"
    data = pd.read_excel(path, engine="openpyxl")
    freq_error = data.freq_error
    cal_error = data.cal_error
    err_freq = cal_abs_mean(freq_error)
    err_cal = cal_abs_mean(cal_error)
    print(err_freq, err_cal)


def plot_heart():
    path = r"D:\my_projects_V1\my_projects\PPG_V1\results\paper_test\hr\hr_new_modify.xlsx"
    data = pd.read_excel(path, engine="openpyxl")
    real = data.real
    freq_cal = data.freq_cal
    cal = data.int_value

    x = np.arange(45, 130)
    y = x

    y3 = 0.98 * x
    y4 = 1.02 * x
    # fig, ax = plt.subplots()
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot()
    ax.plot(x, y, color="slategrey", label="误差=0", linestyle=':')
    ax.plot(x, y3, color="lightcoral", label="误差=2%", linestyle=(0, (3, 1, 1, 1, 1, 1)))
    ax.plot(x, y4, color="lightcoral", linestyle=(0, (3, 1, 1, 1, 1, 1)))
    ax.scatter(real, cal, color="darkviolet", label="样本点", marker="*")
    # plt.scatter(real, freq_cal, color="navy", label="频域方式", marker=7)
    # ax.set_title("心率计算误差分析")
    ax.set_xlabel("计算心率值(次/min)")
    ax.set_ylabel("真实心率值(次/min)")

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # ax.spines['bottom'].set_visible(False)
    # ax.spines['left'].set_visible(False)
    plt.legend(loc="best")
    plt.savefig(os.path.join(PAPER_FIGURE_PATH, "心率误差分析"), dpi=300, bbox_inches="tight")
    plt.show()

def plot_freq():
    path = r"D:\my_projects_V1\my_projects\PPG_V1\results\paper_test\hr\hr_new_modify.xlsx"
    data = pd.read_excel(path, engine="openpyxl")
    real = data.real
    freq_cal = data.freq_cal
    cal = data.int_value

    x = np.arange(45, 130)
    y = x
    y1 = 0.98 * x
    y2 = 1.02 * x

    y3 = 0.98 * x
    y4 = 1.02 * x
    # fig, ax = plt.subplots()
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot()
    ax.plot(x, y, color="slategrey", label="误差=0", linestyle=':')
    ax.plot(x, y1, color="red", label="误差=3%", linestyle='--')
    ax.plot(x, y2, color="red", linestyle='--')
    # ax.plot(x, y3, color="magenta", linestyle=(0, (3, 1, 1, 1, 1, 1)))
    # ax.plot(x, y4, color="magenta", linestyle=(0, (3, 1, 1, 1, 1, 1)))
    ax.scatter(real, cal, color="deepskyblue", label="本文算法", marker="*")
    # plt.scatter(real, freq_cal, color="navy", label="频域方式", marker=7)
    ax.set_title("本文算法误差分析")
    ax.set_xlabel("计算心率值(次/min)")
    ax.set_ylabel("真实心率值(次/min)")

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # ax.spines['bottom'].set_visible(False)
    # ax.spines['left'].set_visible(False)
    plt.legend(loc="best")
    # plt.savefig(os.path.join(PAPER_FIGURE_PATH, "心率误差分析"), dpi=300, bbox_inches="tight")
    plt.show()



if __name__ == '__main__':
    # main()
    # test()
    # abs_mean()
    plot_heart()
    # plot_freq()