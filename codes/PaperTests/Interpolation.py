# -*- coding: utf-8 -*-

"""
@author: Mark Wu
@file: Interpolation.py
@time: 2023/1/27 10:31
"""

import numpy as np
import os
from copy import deepcopy
from scipy import interpolate
import matplotlib.pyplot as plt
from codes.utils.GetFileData import read_from_file
from codes.PaperTests.peakTest import peakTest
from codes.utils.MyFilters import reverse, bandpass_filter
# from codes.utils.Normal import normalization
from codes.PaperTests import PAPER_FIGURE_PATH

plt.rcParams['font.sans-serif'] = ['SongNTR']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

def getData(file_path):
    data = read_from_file(file_path)
    return data

def getPoints(data):
    peaks = peakTest(data)
    reverse_data = reverse(data)
    # peak_values = [data[i] for i in peaks]
    valleys = peakTest(reverse_data)
    # valley_values = [data[i] for i in valleys]
    # plt.figure(figsize=(10, 8))
    # plt.subplot(211)
    # plt.plot(data)
    # plt.scatter(peaks, values, marker="*")
    # plt.plot(data, color="red")
    # plt.plot(data, color="green")
    # plt.scatter(valleys, valley_values, marker="*")
    # plt.show()

    # peaks = []
    # valleys = []
    return peaks, valleys

def deal(data):
    data = bandpass_filter(data, start_fs=0.1, end_fs=5)
    return data

def interpolation(data, points):
    '''
    三次样条插值实现
    '''
    length = len(data)
    lenpoints = len(points)

    temp_points = deepcopy(points)
    temp_points.insert(0, 0)
    temp_points.append(len(data)-1)

    Y = [data[x] for x in temp_points]

    inpolate_points = np.arange(0, length)
    para = interpolate.splrep(temp_points, Y, k=3)
    inpvalue = interpolate.splev(inpolate_points, para)

    return (inpolate_points, inpvalue)

def main():
    file_path = r'D:\my_projects_V1\my_projects\pyqt\learnDemo\data\2022_5_3_15_48.txt'
    data = getData(file_path)
    ir2 = data.ir2
    ir2 = ir2[0:8000]
    predata = deal(ir2)
    peaks, valleys = getPoints(predata)
    peak_values = [predata[y] for y in peaks]
    valley_values = [predata[i] for i in valleys]
    print(len(valleys), len(valley_values))

    inpolate_points, inpvalue = interpolation(predata, peaks)
    valley_inpolate_points, valley_inpvalue = interpolation(predata, valleys)

    mean_value = [(inpvalue[i]+valley_inpvalue[i]) / 2 for i in range(len(valley_inpolate_points))]

    new_data1 = [(predata[i] - valley_inpvalue[i]) for i in range(len(valley_inpvalue))]
    new_data2 = [(predata[i] - mean_value[i]) for i in range(len(mean_value))]


    # 包络线
    # plt.plot(inpolate_points, inpvalue)
    # plt.plot(valley_inpolate_points, valley_inpvalue)
    # plt.show()

    new_peaks, new_valleys = getPoints(new_data1)
    peaks_y = [new_data1[x] for x in new_peaks]
    diff = np.diff(peaks)
    valley_diff = np.diff(new_peaks)
    # print(diff)
    # print(valley_diff)
    # print(np.mean(diff), np.mean(valley_diff))

    plt.figure(figsize=(10, 8))
    # plt.subplot(211)
    # plt.plot(data)
    # plt.scatter(peaks, values, marker="*")
    # plt.plot(data, color="red")
    plt.plot(predata, color="lightseagreen", label="PPG")
    plt.scatter(valleys, valley_values, marker="*", label="波谷点", color="orange")
    plt.plot(valley_inpolate_points, valley_inpvalue, color='fuchsia', label="三次样条插值曲线")
    plt.xlabel("采样点")
    plt.ylabel("幅值")
    plt.legend(loc="best")
    plt.savefig(os.path.join(PAPER_FIGURE_PATH, "三次样条插值图"), dpi=300, bbox_inches="tight")
    plt.show()

    plt.figure(figsize=(10, 8))
    plt.subplot(211)
    plt.plot(predata, color='mediumpurple', label="PPG")
    plt.scatter(peaks, peak_values, color="deeppink", label="波峰点")
    plt.title("三次样条插值前的PPG")
    plt.legend(loc="best")
    # plt.scatter(inpolate_points, inpvalue, marker="*")
    plt.subplot(212)
    plt.plot(new_data1, color="cyan", label="PPG")
    plt.scatter(new_peaks, peaks_y, color="slateblue", label="波峰点")
    plt.title("三次样条插值后的PPG")
    plt.xlabel("采样点")
    # plt.plot(new_data2, color="green")
    # plt.xticks(fontsize=10.5)
    # plt.yticks(fontsize=10.5)
    plt.legend(loc="best")
    plt.savefig(os.path.join(PAPER_FIGURE_PATH, "三次样条插值对比图"), dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == '__main__':
    main()
