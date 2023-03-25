# -*- coding: utf-8 -*-

"""
@author: Mark Wu
@file: ButterFilter.py
@time: 2023/2/8 11:57
"""

import os
import matplotlib.pyplot as plt
from codes.utils.GetFileData import read_from_file
from codes.utils.MyFilters import bandpass_filter
from codes.PaperTests import PAPER_FIGURE_PATH

plt.rcParams["font.sans-serif"]=["SimHei"] #设置字体
plt.rcParams["axes.unicode_minus"]=False #正常显示负号

def get_data(path):
    data = read_from_file(path)
    ir2 = data.ir2
    return ir2

def butter():
    path = r'D:\my_projects_V1\my_projects\pyqt\learnDemo\data\2022_5_3_15_48.txt'
    data = get_data(path)[1000: 5000]
    buttered = bandpass_filter(data, start_fs=0.1, end_fs=5)
    plt.figure(figsize=(10, 8))
    plt.plot(data, label="PPG")
    plt.title("原始PPG信号")
    plt.ylabel("采样点")
    plt.xlabel("幅值")
    plt.legend(loc="best")
    plt.savefig(os.path.join(PAPER_FIGURE_PATH, "butter原始信号"), dpi=300)
    plt.show()

    plt.figure(figsize=(10, 8))
    plt.plot(buttered, label="filtered")
    plt.title("滤波后的PPG信号")
    plt.ylabel("幅值")
    plt.xlabel("采样点")
    plt.legend(loc="best")
    plt.savefig(os.path.join(PAPER_FIGURE_PATH, "butter滤波后"), dpi=200)
    plt.show()

if __name__ == '__main__':
    butter()

