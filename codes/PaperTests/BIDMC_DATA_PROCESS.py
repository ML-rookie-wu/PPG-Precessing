# -*- coding: utf-8 -*-

"""
@author: Mark Wu
@file: BIDMC_DATA_PROCESS.py
@time: 2023/2/26 fast:45
"""
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
import math
from codes.utils.MyFilters import vmd, bandpass_filter, reverse
from codes.utils.Normal import normalization
from codes.utils.GetFFT import signal_fft, get_freq
from codes.utils.SaveToExcel import save_to_excel
from codes.PaperTests import PAPER_FIGURE_PATH
from codes.PaperTests.CalculateError import cal_mse


plt.rcParams['font.sans-serif'] = ['SongNTR']
plt.rcParams['axes.unicode_minus'] = False


def peak_detect(filtered_data, interval):
    peakList = []
    ir2_max = np.max(filtered_data)
    ir2_min = np.min(filtered_data)
    ir2_mean = (sum(filtered_data) - ir2_max - ir2_min) / (len(filtered_data) - 2)

    diff = np.diff(filtered_data)
    diff_max = np.max(diff)
    diff_min = np.min(diff)
    diff_mean = (sum(diff) - diff_max - diff_min) / (len(diff) - 2)

    th1 = 1.2 * diff_mean
    normal_ir2 = normalization(filtered_data)
    normal_diff = normalization(diff)
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
                        if peak_index - peakList[-1] < 55:
                            break
                        else:
                            peakList.append(peak_index)
    return peakList

def get_data(file_path):
    data = pd.read_csv(file_path)
    ir = data[" PLETH"]
    resp = data[" RESP"]
    return ir, resp

def get_record(record_path):
    record = pd.read_csv(record_path)
    HR = record[" HR"]
    PULSE = record[" PULSE"]
    RESP = record[" RESP"]
    return HR, PULSE, RESP


def cal_one_data(dir_path, num):
    # frequence = 125
    if num < 10:
        index = "0%s" % num
    else:
        index = str(num)
    name = "Signals"
    record = "Numerics"
    file_name = "bidmc_%s_%s.csv" % (index, name)
    record_name = "bidmc_%s_%s.csv" % (index, record)
    file_path = os.path.join(dir_path, file_name)
    print(file_path)
    record_path = os.path.join(dir_path, record_name)
    hr_record, pulse_record, resp_record = get_record(record_path)
    # print(resp_record)
    real_pulse = pulse_record[1]
    real_resp = resp_record[1]

    ir, resp = get_data(file_path)
    ir_temp = ir[0:7500]
    ir_data = reverse(ir_temp)

    # resp_data = resp[0:7500]
    # resp_butter = bandpass_filter(resp_data, fs=125, start_fs=0.1, end_fs=0.7)
    # f0, absY0 = signal_fft(resp_butter, freq=125)
    # freq0, amp0 = get_freq(f0, absY0)
    # print("freq0=", freq0)

    ir_butter = bandpass_filter(ir_data, fs=125, start_fs=0.5, end_fs=3)
    peaks_list = peak_detect(ir_butter)
    peaks_value = [ir_butter[i] for i in peaks_list]

    ppTime = np.diff(peaks_list)
    intervel_max = max(ppTime)
    intervel_min = min(ppTime)
    # print(ppTime)
    hr = 125 / ((np.sum(ppTime) - intervel_max - intervel_min) / (len(ppTime) - 2)) * 60

    result, u, vmd_resp = vmd(ir_temp)
    vmd_resp_butter = bandpass_filter(vmd_resp, fs=125, start_fs=0.24, end_fs=0.8)
    f, absY = signal_fft(vmd_resp_butter, freq=125)
    freq, max_amp = get_freq(f, absY)
    # plt.subplot(211)
    # plt.plot(f, absY)
    # plt.subplot(212)
    # plt.plot(vmd_resp_butter)
    # plt.show()
    cal_freq = round(freq*60, 0)
    peaks_num = len(peaks_list)

    print(real_pulse, real_resp, peaks_num, round(hr, 0), cal_freq)
    return [real_pulse, peaks_num, real_resp, cal_freq]

def plot_pulse():
    path = r'D:\my_projects_V1\my_projects\PPG_V1\results\paper_test\bidmc_results_final.xlsx'
    results = pd.read_excel(path, engine="openpyxl")
    real_pulse = results.real_pulse
    cal_pulse = results.cal_pulse
    x = np.arange(40, 130)
    y = x
    y1 = 0.95 * x
    y2 = 1.05 * x
    # fig, ax = plt.subplots()
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot()
    ax.plot(x, y, color="fuchsia", label="误差=0", linestyle=':')
    ax.plot(x, y1, color="red", label="误差=5%", linestyle='--')
    ax.plot(x, y2, color="red", linestyle='--')
    ax.scatter(cal_pulse, real_pulse, color="cyan", label="数据点")
    ax.set_title("脉率误差分析")
    ax.set_xlabel("计算脉率值(次/min)")
    ax.set_ylabel("真实脉率值(次/min)")

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # ax.spines['bottom'].set_visible(False)
    # ax.spines['left'].set_visible(False)
    plt.legend(loc="best")
    plt.savefig(os.path.join(PAPER_FIGURE_PATH, "脉率误差分析"), dpi=300, bbox_inches="tight")
    plt.show()

def plot_resp():
    path = r'D:\my_projects_V1\my_projects\PPG_V1\results\paper_test\bidmc_results_final.xlsx'
    results = pd.read_excel(path, engine="openpyxl")
    real_resp = results.real_resp
    cal_resp = results.cal_resp
    print(cal_resp)
    x = np.arange(10, 30)
    y = x
    y1 = 0.9 * x
    y2 = 1.1 * x
    # fig, ax = plt.subplots()
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot()
    ax.plot(x, y, color="deeppink", label="误差=0", linestyle=':')
    ax.plot(x, y1, color="red", label="误差=10%", linestyle='--')
    ax.plot(x, y2, color="red", linestyle='--')
    ax.scatter(cal_resp, real_resp, color="blueviolet", label="数据点")
    ax.set_title("呼吸频率误差分析")
    # ax.set_xlim(10, 30)
    ax.set_xticks(range(10, 35, 5))
    ax.set_xlabel("计算呼吸频率值(次/min)")
    ax.set_ylabel("真实呼吸频率值(次/min)")

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # ax.spines['bottom'].set_visible(False)
    # ax.spines['left'].set_visible(False)
    plt.legend(loc="best")
    plt.savefig(os.path.join(PAPER_FIGURE_PATH, "呼吸频率误差分析"), dpi=300, bbox_inches="tight")
    plt.show()

def mse():
    path = r'D:\my_projects_V1\my_projects\PPG_V1\results\paper_test\bidmc_results_final.xlsx'
    results = pd.read_excel(path, engine="openpyxl")
    resp_error = results.resp_error
    pulse_error = results.pulse_error
    mse_resp = cal_mse(resp_error)
    mse_pulse = cal_mse(pulse_error)
    print(mse_resp, mse_pulse)

def main():
    dir_path = r"D:\my_projects_V1\my_projects\BIDMC_DATA\bidmc-dataset-1.0.0\bidmc_csv"
    results = []
    for i in range(1, 54):
        cal_list = cal_one_data(dir_path, i)
        results.append(cal_list)
    save_path = r'D:\my_projects_V1\my_projects\PPG_V1\results\paper_test\bidmc_results.xlsx'
    column_name = ["real_pulse", "cal_pulse", "real_resp", "cal_resp"]
    save_to_excel(save_path, results, column_name)


if __name__ == '__main__':
    # main()
    # plot_pulse()
    plot_resp()
    # mse()