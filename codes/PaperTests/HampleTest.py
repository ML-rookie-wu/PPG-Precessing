# -*- coding: utf-8 -*-

"""
@author: Mark Wu
@file: HampleTest.py
@time: 2023/2/7 10:24
"""


import numpy as np
import os
import matplotlib.pyplot as plt
from codes.utils.MyFilters import bandpass_filter
from codes.utils.GetFileData import read_from_file
from codes.PaperTests import PAPER_FIGURE_PATH


plt.rcParams["font.sans-serif"]=["SimHei"] #设置字体
plt.rcParams["axes.unicode_minus"]=False #正常显示负号


def reverse(data):
    """
    反转波形
    """
    data_max = max(data)
    reversed_data = [data_max - _ for _ in data]
    return reversed_data

def hampel(X, k):
    length = X.shape[0] - 1
    nsigma = 3
    iLo = np.array([i - k for i in range(0, length + 1)])
    iHi = np.array([i + k for i in range(0, length + 1)])
    iLo[iLo < 0] = 0
    iHi[iHi > length] = length
    xmad = []
    xmedian = []
    for i in range(length + 1):
        w = X[iLo[i]:iHi[i] + 1]
        medj = np.median(w)
        mad = np.median(np.abs(w - medj))
        xmad.append(mad)
        xmedian.append(medj)
    xmad = np.array(xmad)
    xmedian = np.array(xmedian)
    scale = 1.4826  # 缩放
    xsigma = scale * xmad
    xi = ~(np.abs(X - xmedian) <= nsigma * xsigma)  # 找出离群点（即超过nsigma个标准差）

    # 将离群点替换为中为数值
    xf = X.copy()
    xf[xi] = xmedian[xi]
    return xf

def hample_test():
    X = np.array([1, 2, 3, 4, 100, 4, 3, 2, 1])
    res = hampel(X, 3)
    plt.plot(X, label="原始数据")
    plt.plot(res, '--', label="去除异常值后的数据")
    plt.legend(loc="best")
    plt.savefig(os.path.join(PAPER_FIGURE_PATH, "hample_test"), dpi=200)
    plt.show()


def hample_ppg():
    path = r'D:\my_projects_V1\my_projects\PPG_V1\data\paper_picture\hample_test.txt'
    data = read_from_file(path)
    ir2 = data.ir2
    # ir2 = reverse(data.ir2)
    # filtered_ir2 = bandpass_filter(ir2, start_fs=0.1, end_fs=5)
    hampled = hampel(ir2, 3)
    plt.figure(figsize=(10, 8))
    plt.scatter(9998, ir2[9998], marker="o", color="red", label="outlier")
    plt.plot(ir2, label="PPG")
    plt.ylabel("幅值")
    plt.xlabel("采样点")
    plt.title("原始PPG信号")
    plt.legend(loc="best")
    plt.savefig(os.path.join(PAPER_FIGURE_PATH, "异常值原始数据"), dpi=200)
    plt.show()


    plt.figure(figsize=(10, 8))
    plt.plot(ir2[9000: 11000])
    plt.savefig(os.path.join(PAPER_FIGURE_PATH, "异常值放大"))
    plt.show()


    plt.figure(figsize=(10, 8))
    plt.plot(hampled, label="PPG")
    plt.xlabel("采样点")
    plt.ylabel("幅值")
    plt.title("去除异常值后的PPG")
    plt.legend(loc="best")
    plt.savefig(os.path.join(PAPER_FIGURE_PATH, "使用hample后"), dpi=200)
    plt.show()


def main():
    hample_test()
    # hample_ppg()


if __name__ == '__main__':
    main()
