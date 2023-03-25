# -*- coding: utf-8 -*-

"""
@author: Mark Wu
@file: WaveletTest.py
@time: 2023/2/15 13:30
"""
import matplotlib.pyplot as plt
import os
from codes.utils.MyFilters import get_dwt_res, bandpass_filter
from codes.utils.GetFileData import read_from_file
from codes.utils.GetFFT import get_freq, signal_fft
from codes.PaperTests import PAPER_FIGURE_PATH


plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

def get_data(path):
    data = read_from_file(path)
    return data

def get_dwt(data):
    rec_a, rec_d = get_dwt_res(data)

    fig = plt.figure(figsize=(16, 12))
    ax_main = fig.add_subplot(len(rec_a) + 1, 1, 1)
    # ax_main.set_title(title)
    ax_main.plot(data)
    ax_main.set_title("10层小波分解")
    ax_main.set_xlim(0, len(data) - 1)

    for i, y in enumerate(rec_a):
        # print(len(rec_a[i]))
        ax = fig.add_subplot(len(rec_a) + 1, 2, 3 + i * 2)
        ax.plot(y, color="cyan")
        ax.set_xlim(0, len(y) - 1)
        ax.set_ylabel("A%d" % (i + 1))
        plt.subplots_adjust(hspace=0.5)

    for i, y in enumerate(rec_d):
        # print(len(rec_d[i]))
        ax = fig.add_subplot(len(rec_d) + 1, 2, 4 + i * 2)
        ax.plot(y, color="deeppink")
        ax.set_xlim(0, len(y) - 1)
        ax.set_ylabel("D%d" % (i + 1))
        plt.subplots_adjust(hspace=0.5)
    # plt.savefig(os.path.join(PAPER_FIGURE_PATH, "小波分解ppg图"), dpi=300)
    # plt.show()
    return rec_a, rec_d

def freq_analy(data, rec_a, rec_d):
    fig = plt.figure(figsize=(16, 12))
    ax_main = fig.add_subplot(len(rec_a) + 1, 1, 1)
    # ax_main.set_title(title)
    ax_main.plot(data)
    ax_main.set_title("10层小波分解")
    ax_main.set_xlim(0, len(data) - 1)

    for i, y in enumerate(rec_a):
        ax = fig.add_subplot(len(rec_a) + 1, 2, 3 + i * 2)
        ax.plot(y, color="cyan")
        ax.set_xlim(0, len(y) - 1)
        ax.set_ylabel("A%d" % (i + 1))
        plt.subplots_adjust(hspace=0.5)

    # for i, y in enumerate(rec_d):
    #
    #     ax = fig.add_subplot(len(rec_d) + 1, 2, 3 + i * 2)
    #     ax.plot(y, 'r')
    #     ax.set_xlim(0, len(y) - 1)
    #     ax.set_ylabel("A%d" % (i + 1))
    #     plt.subplots_adjust(hspace=0.5)

    for i, y in enumerate(rec_a):
        ax = fig.add_subplot(len(rec_a) + 1, 2, 4 + i * 2)
        y_ppg, abs_y_ppg = signal_fft(rec_a[i], freq=500)
        ax.plot(y_ppg, abs_y_ppg, color="limegreen")
        # ax.set_xlim(0, len(rec_a[i]) - 1)
        ax.set_ylabel("F%d" % (i + 1))
        plt.subplots_adjust(hspace=0.5)

    # plt.figure(figsize=(10, 8))
    #
    # for i in range(len(rec_a)):
    #     plt.subplot(len(imfs), 1, i + 1)
    #     y_ppg, abs_y_ppg = signal_fft(imfs[i], freq)
    #
    #     freq, max_ap = get_freq(y_ppg, abs_y_ppg)
    #     max_freq_list.append((freq, max_ap))
    #     plt.plot(y_ppg, abs_y_ppg)
    #     plt.ylabel("D%d" % (i + 1))
    #     plt.subplots_adjust(hspace=0.5)
    # plt.show()
    # plt.savefig(os.path.join(PAPER_FIGURE_PATH, "小波变换频谱图1"))
    plt.show()


def main():
    path = r'D:\my_projects_V1\my_projects\PPG_V1\data\spo2_compare_new\3disturb_3pulse\98\20221130141715_98.txt'
    data = get_data(path)
    ir2 = data.ir2
    butter = bandpass_filter(ir2, start_fs=0.1, end_fs=5)
    rec_a, rec_d = get_dwt(butter)
    # rec_a, rec_d = get_dwt(ir2)
    freq_analy(butter, rec_a, rec_d)
    # freq_analy(ir2, rec_a, rec_d)


if __name__ == '__main__':
    main()

