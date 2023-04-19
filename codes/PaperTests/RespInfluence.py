# -*- coding: utf-8 -*-

"""
@author: Mark Wu
@file: RespInfluence.py
@time: 2023/2/26 16:49
"""


import matplotlib.pyplot as plt
import numpy as np
import os
from copy import deepcopy
from scipy import interpolate
from codes.utils.Normal import normalization
from codes.utils.GetFileData import read_from_file
from codes.utils.MyFilters import bandpass_filter, reverse
from codes.PaperTests import PAPER_FIGURE_PATH


# plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['font.sans-serif'] = ['SongNTR']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


def get_data(path):
    data = read_from_file(path)
    return data.ir2

def peak_detect(filtered_data):
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
                        if peak_index - peakList[-1] < 250:
                            break
                        else:
                            peakList.append(peak_index)
    return peakList

def interpolation(data, points):
    '''
    三次样条插值实现
    '''
    length = len(data)
    lenpoints = len(points)

    temp_points = deepcopy(points)

    Y = [data[x] for x in temp_points]

    inpolate_points = np.arange(0, length)
    para = interpolate.splrep(temp_points, Y, k=3)
    inpvalue = interpolate.splev(inpolate_points, para)

    return (inpolate_points, inpvalue)

def pav(data):
    buttered = bandpass_filter(data, start_fs=0.1, end_fs=4)
    revesed = reverse(buttered)
    valleys = peak_detect(revesed)
    peaks = peak_detect(buttered)
    peaks_value = [buttered[i] for i in peaks]
    valleys_value = [buttered[x] for x in valleys]
    plt.figure(figsize=(10, 8))
    plt.plot(buttered, color='royalblue', label="PPG")
    plt.title("PAV")
    plt.xlabel("采样点")
    plt.ylabel("幅值")
    plt.scatter(peaks, peaks_value, color="red", label="波峰点")
    plt.scatter(valleys, valleys_value, color="darkorchid", label="波谷点")
    plt.legend(loc="best")
    plt.savefig(os.path.join(PAPER_FIGURE_PATH, "振幅调制2"), dpi=300, bbox_inches="tight")
    plt.show()

def riiv(data):
    buttered = bandpass_filter(data, start_fs=0.1, end_fs=4)
    revesed = reverse(buttered)
    valleys = peak_detect(revesed)
    peaks = peak_detect(buttered)
    peaks_value = [buttered[i] for i in peaks]
    valleys_value = [buttered[x] for x in valleys]

    interp_points, interp_values = interpolation(buttered, valleys)
    plt.figure(figsize=(10, 8))
    plt.plot(buttered, color='royalblue', label="PPG")
    plt.title("RIIV")
    plt.xlabel("采样点")
    plt.ylabel("幅值")
    plt.scatter(valleys, valleys_value, color="darkorange", label="波谷点")
    plt.plot(interp_points, interp_values, color="red", label="基线调制")
    plt.legend(loc="best")
    plt.savefig(os.path.join(PAPER_FIGURE_PATH, "基线调制2"), dpi=300, bbox_inches="tight")
    plt.show()

def prv(data):
    buttered = bandpass_filter(data, start_fs=0.1, end_fs=5)[4300:8000]
    revesed = reverse(buttered)
    valleys = peak_detect(revesed)
    peaks = peak_detect(buttered)
    peaks_value = [buttered[i] for i in peaks]
    valleys_value = [buttered[x] for x in valleys]
    plt.figure(figsize=(10, 8))
    plt.plot(buttered, color='royalblue', label="PPG")
    plt.title("PRV")
    plt.xlabel("采样点")
    plt.ylabel("幅值")
    plt.scatter(peaks, peaks_value, color="darkorange", label="波谷点")
    plt.legend(loc="best")
    plt.savefig(os.path.join(PAPER_FIGURE_PATH, "频率调制2"), dpi=300, bbox_inches="tight")
    plt.show()
def main():
    path = r'D:\my_projects_V1\my_projects\PPG_V1\data\paper_test\1disturb_3pulse\98\20221129195154_98.txt'
    # path = r'D:\my_projects_V1\my_projects\PPG_V1\data\hr_test\0disturb\90\20230209103108_90.txt'
    # path = r'D:\my_projects_V1\my_projects\PPG_V1\data\real_data\paper_resp\2023_02_17_12_02_54.txt'

    ir2 = get_data(path)
    ir2_pav = ir2[0:4000]
    ir2_riiv = ir2[0:15000]
    pav(ir2_pav)
    riiv(ir2_riiv)
    prv(ir2_riiv)



if __name__ == '__main__':
    main()

