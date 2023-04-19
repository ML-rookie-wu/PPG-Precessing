# -*- coding: utf-8 -*-

"""
@author: Mark Wu
@file: EMDTest.py
@time: 2023/1/31 11:29
"""

import matplotlib.pyplot as plt
from PyEMD import EMD, Visualisation
import numpy as np
import os
from codes.utils.GetFFT import signal_fft, get_freq
from codes.utils.GetFileData import read_from_file
from codes.utils.MyFilters import bandpass_filter
from codes.PaperTests import PAPER_FIGURE_PATH

plt.rcParams['font.sans-serif'] = ['SongNTR']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

def getData(file_path):
    data = read_from_file(file_path)
    return data

def MyEMD(data):
    emd = EMD()
    emd.emd(data)
    imfs, res = emd.get_imfs_and_residue()
    return imfs, res

def emd_save_plot(data):
    s = np.array(data)
    imfs, res = MyEMD(s)
    fig = plt.figure(figsize=(16, 10))
    ax_main = fig.add_subplot(len(imfs) + 3, 1, 1)
    ax_main.plot(data, color="magenta")
    ax_main.set_title("原始PPG信号")
    ax_main.set_xlim(0, len(data) - 1)

    # ax1 = fig.add_subplot(len(imfs)+4, 1, 2)
    # ax1.plot(imfs[0])
    # ax1.set_title("EMD分解的分量")
    # plt.subplots_adjust(hspace=0.9)

    for i in range(len(imfs)):
        ax = fig.add_subplot(len(imfs) + 4, 1, i+3)
        ax.plot(imfs[i], color="turquoise")
        if i == 0:
            ax.set_title("EMD分解的分量")
        ax.set_xlim(0, len(imfs[i]) - 1)
        # ax.set_ylabel("imf%d" % (i + 1), fontsize=10.5)
        plt.subplots_adjust(hspace=0.9)

    ax = fig.add_subplot(len(imfs)+4, 1, len(imfs)+4)
    ax.plot(res, color="darkred")
    ax.set_title("残差")
    # ax.set_ylabel("residual")
    plt.savefig(os.path.join(PAPER_FIGURE_PATH, "emd分解图"), dpi=300, bbox_inches="tight")
    plt.show()


def emd_plot(data, freq=500):
    s = np.array(data)
    imfs, res = MyEMD(s)

    max_freq_list = list()
    # plt.figure(figsize=(10, 8))
    # for i in range(len(imfs)):
    #     plt.subplot(len(imfs), 1, i+1)
    #     y_ppg, abs_y_ppg = signal_fft(imfs[i], freq)
    #
    #     freq, max_ap = get_freq(y_ppg, abs_y_ppg)
    #     max_freq_list.append((freq, max_ap))
    #     # plt.plot(y_ppg, abs_y_ppg)
    #     plt.plot(imfs[i])
    #     plt.ylabel("imf%d" % (i + 1))
    #     plt.subplots_adjust(hspace=0.5)
    # plt.show()

    plt.plot(s)
    plt.title("ppg")
    plt.xlabel("采样点")
    plt.ylabel("幅值")
    plt.show()

    fig = plt.figure(figsize=(16, 12))
    # ax_main = fig.add_subplot(len(imfs) + 1, 1, 1)
    # ax_main.set_title(title)
    # ax_main.plot(s, color="green")
    # ax_main.set_xlim(0, len(data) - 1)


    for i, y in enumerate(imfs):

        plt.subplot(len(imfs), 1, i+1)
        if i == 0:
            plt.title("IMFs")
        if i == len(imfs)-1:
            plt.xlabel("采样点")
        plt.plot(y)
        plt.xlim(0, len(y)-1)
        # ax = fig.add_subplot(len(imfs), 1, i+1)
        # ax.plot(y)
        # ax.set_xlim(0, len(y) - 1)
        plt.ylabel("imf%d" % (i + 1))
        plt.subplots_adjust(hspace=0.8)

    # # 绘图 IMF
    # vis = Visualisation()
    # vis.plot_imfs(imfs=imfs, residue=res, t=t, include_residue=True)
    # # 绘制并显示所有提供的IMF的瞬时频率
    # vis.plot_instant_freq(t, imfs=imfs)
    # vis.show()
    # plt.title("IMFs")
    plt.show()

    return imfs, max_freq_list

def emd_freq_test(data):
    # t = np.arange(0, end, 0.01)
    s = np.array(data)
    imfs, res = MyEMD(s)

    max_freq_list = list()
    plt.figure(figsize=(10, 8))
    for i in range(len(imfs)):
        plt.subplot(len(imfs), 1, i+1)
        y_ppg, abs_y_ppg = signal_fft(imfs[i], freq=500)
        freq, max_ap = get_freq(y_ppg, abs_y_ppg)
        max_freq_list.append((freq, max_ap))
        # plt.plot(y_ppg, abs_y_ppg)
        plt.plot(y_ppg, abs_y_ppg)
        plt.ylabel("imf%d" % (i + 1))
        plt.subplots_adjust(hspace=0.5)
    plt.show()

def deal(data):
    data = bandpass_filter(data, start_fs=0.1, end_fs=5)
    return data

def emd_test(data):
    s = np.array(data)
    butter = bandpass_filter(data, start_fs=0.1, end_fs=5)
    imfs, res = MyEMD(s)
    restruct_list = []
    for i in range(len(imfs)):
        y_ppg, abs_y_ppg = signal_fft(imfs[i], freq=500)
        freq, max_ap = get_freq(y_ppg, abs_y_ppg)
        if 0.1 < freq < 2:
            restruct_list.append(imfs[i])
    print(len(restruct_list))
    restruct_data = np.array(restruct_list[0])
    if len(restruct_list) > 1:
        for i in range(1, len(restruct_list)):
            restruct_data += np.array(restruct_list[i])
    plt.figure(figsize=(10, 8))
    plt.subplot(311)
    plt.plot(s)
    plt.title("原始信号")
    plt.subplot(312)
    plt.plot(restruct_data)
    plt.title("emd重构")
    plt.subplot(313)
    plt.plot(butter)
    plt.title("butterworth滤波")
    plt.show()

def emd_tese1(data):
    s = np.array(data)
    # butter = bandpass_filter(data, start_fs=0.1, end_fs=5)
    imfs, res = MyEMD(s)
    restructed = sum(imfs)
    plt.figure(figsize=(10, 8))
    plt.subplot(211)
    plt.plot(s)
    plt.subplot(212)
    plt.plot(restructed)
    plt.show()


def main():
    # path = r'D:\my_projects_V1\my_projects\pyqt\learnDemo\data\2022_5_3_15_48.txt'
    path = r'D:\my_projects_V1\my_projects\PPG_V1\data\spo2_compare_new\1disturb_3pulse\98\20221129195154_98.txt'
    data = getData(path)
    ir2 = data.ir2
    partial_data = ir2[0:8000]
    # imfs, max_freq_list = emd_plot(ir2)
    emd_save_plot(partial_data)
    # emd_freq_test(ir2)
    # emd_test(ir2)
    # emd_tese1(ir2)

if __name__ == '__main__':
    main()

