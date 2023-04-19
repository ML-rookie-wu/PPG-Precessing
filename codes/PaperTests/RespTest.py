# -*- coding: utf-8 -*-

"""
@author: Mark Wu
@file: RespTest.py
@time: 2023/2/17 13:08
"""
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pylab as pl

from codes.utils.GetFileData import travel_dir, read_from_file
from codes.utils.MyFilters import vmd, bandpass_filter
from codes.utils.GetFFT import signal_fft, get_freq
from codes.PaperTests import PAPER_FIGURE_PATH
from codes.utils.SaveToExcel import save_to_excel
from codes.utils.Normal import normalization

plt.rcParams['font.sans-serif'] = ['SongNTR']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


def get_data(path):
    data = read_from_file(path)
    return data

def resp_compare():

    path = r'D:\my_projects_V1\my_projects\PPG_V1\data\real_data\paper_resp\2023_02_17_12_45_36.txt'
    # all_files = travel_dir(dir)
    data = get_data(path)
    real_resp = data.resp
    real_resp = bandpass_filter(real_resp, start_fs=0.1, end_fs=0.6)
    f_real_resp, real_absY = signal_fft(real_resp)
    ir2 = data.ir2
    # ir2 = bandpass_filter(ir2, start_fs=0.1, end_fs=5)

    filtered, imfs, resp = vmd(ir2)
    resp = bandpass_filter(resp, start_fs=0.2, end_fs=0.6)
    f, absY = signal_fft(resp)
    plt.figure(figsize=(10, 8))
    plt.subplot(411)
    plt.plot(real_resp)
    plt.subplot(412)
    plt.plot(f_real_resp, real_absY)
    plt.subplot(413)
    plt.plot(resp)
    plt.subplot(414)
    plt.plot(f, absY)
    plt.show()

def cal_resp():
    dir = r'D:\my_projects_V1\my_projects\PPG_V1\data\resp_test'
    all_files = travel_dir(dir)
    resp_results = []
    for file_path in all_files:
        print(file_path)
        real_resp = os.path.split(os.path.dirname(file_path))[1]
        data = read_from_file(file_path)
        ir2 = data.ir2
        start = 0
        window = 15000
        step = 3000
        end = start + window

        while end < len(ir2):

            filtered, imfs, resp = vmd(ir2[start: end])
            buttered_resp = bandpass_filter(resp, start_fs=0.1, end_fs=0.8)
            f, absY = signal_fft(buttered_resp)
            freq, max_amplitude = get_freq(f, absY)
            cal_resp = freq * 60
            error = int(real_resp) - cal_resp
            print(real_resp, cal_resp, error)
            start += step
            end = start + window
            # resp_results.append([real_resp, cal_resp, error])

    # df = pd.DataFrame(resp_results)
    # column_name = ["real_resp", "cal_resp", "error"]
    # save_path = r'D:\my_projects_V1\my_projects\PPG_V1\results\paper_test\resp\simlator_resp_error.xlsx'
    # save_to_excel(save_path, resp_results, column_name)



def resp_test(path):
    # path = r'D:\my_projects_V1\my_projects\PPG_V1\data\real_data\paper_resp\2023_02_17_12_02_54.txt'
    # path = r'D:\my_projects_V1\my_projects\PPG_V1\data\real_data\paper_resp\2023_02_17_12_45_36.txt'
    # all_files = travel_dir(dir)
    data = get_data(path)
    resp = data.resp
    ir2 = data.ir2
    # ir2 = bandpass_filter(ir2, start_fs=0.1, end_fs=5)

    filtered, imfs, vmd_resp = vmd(ir2)


    fig = plt.figure(figsize=(16, 12))
    ax_main = fig.add_subplot(len(imfs) + 2, 1, 1)
    ax_main.plot(ir2, color="darkviolet")
    ax_main.set_title("原始PPG信号")
    ax_main.set_ylabel("幅值")
    ax_main.set_xlabel("采样点")
    ax_main.set_xlim(0, len(ir2) - 1)

    for i, x in enumerate(imfs):
        ax = fig.add_subplot(len(imfs) + 2, 2, 5 + i * 2)
        if i == 0:
            ax.set_title("VMD分解后的模态分量")
        ax.plot(x, color="hotpink")
        ax.set_xlim(0, len(x) - 1)
        ax.set_ylabel("imf%d" % (i + 1))
        plt.subplots_adjust(hspace=0.8)

    for i, y in enumerate(imfs):
        ax = fig.add_subplot(len(imfs) + 2, 2, 6 + i * 2)
        f, absY = signal_fft(y)
        # freq, max_amp = get_freq(f, absY)
        # if 0.1 < freq < 5:
        #     restruct_list.append(E_IMFs[i])
        if i == 0:
            ax.set_title("模态分量对应的频谱")
        ax.plot(f, absY, color="lightseagreen")
        # ax.set_xlim(0, len(y) - 1)
        ax.set_ylabel("freq%d" % (i + 1))
        plt.subplots_adjust(hspace=0.8)
    # plt.savefig(os.path.join(PAPER_FIGURE_PATH, "vmd分解频谱图"), dpi=300)
    plt.show()

    butter_resp = bandpass_filter(resp, start_fs=0.1, end_fs=0.7)
    resp_freq, resp_amp = signal_fft(butter_resp)
    f1, max_amp1 = get_freq(resp_freq, resp_amp)
    print(f1, max_amp1)
    plt.figure(figsize=(10, 8))
    plt.subplot(211)
    plt.plot(butter_resp, color="gold")
    plt.title("呼吸波信号")
    plt.subplot(212)
    plt.plot(resp_freq, resp_amp, color="aquamarine")
    plt.scatter(f1, max_amp1, marker="*", color="red")
    plt.xlabel("频率")
    plt.title("呼吸波信号频谱")
    pl.show()

    butter_vmd_resp = bandpass_filter(vmd_resp, start_fs=0.2, end_fs=0.7)
    vmd_resp_freq, vmd_resp_amp = signal_fft(butter_vmd_resp)
    f2, max_amp2 = get_freq(vmd_resp_freq, vmd_resp_amp)
    print(f2, max_amp2)
    plt.figure(figsize=(10, 8))
    plt.subplot(211)
    plt.plot(butter_vmd_resp, color="teal")
    plt.title("提取的呼吸信号")
    plt.subplot(212)
    plt.plot(resp_freq, vmd_resp_amp, color="deepskyblue")
    plt.scatter(f2, max_amp2, marker="*", color="red")
    plt.xlabel("频率")
    plt.title("提取的呼吸信号对应频谱")
    pl.show()

    x = normalization(butter_vmd_resp)
    y = normalization(butter_resp)
    print(len(x), len(y))
    up = sum([x[i] * y[i] for i in range(len(x))])/len(x)
    down = np.sqrt(sum([x[i] ** 2 for i in range(len(x))]) / len(x)) * np.sqrt(sum([y[j]**2 for j in range(len(y))]) / len(y))
    print(up/down)
    res = np.cov(x, y) / (np.var(x) * np.var(y))
    print(np.cov(x, y))
    print(np.var(x), np.var(y))
    print(res)

    plt.figure(figsize=(10, 8))
    plt.subplot(211)
    plt.title("原始PPG信号")
    plt.plot(ir2, color="deeppink", label="PPG")
    plt.plot(vmd_resp, color="red", label="基线")
    plt.legend(loc="best")
    plt.subplot(212)
    plt.title("去除基线后的PPG信号")
    plt.xlabel("采样点")
    plt.plot(filtered, color="royalblue", label="PPG")
    plt.legend(loc="best")
    plt.savefig(os.path.join(PAPER_FIGURE_PATH, "vmd分解图"), dpi=300, bbox_inches="tight")
    plt.show()

def main():
    dir = r"D:\my_projects_V1\my_projects\PPG_V1\data\real_data\paper_resp"
    all_files = travel_dir(dir)
    path = r'D:\my_projects_V1\my_projects\PPG_V1\data\real_data\paper_resp\2023_02_17_12_02_54.txt'
    # path = r'D:\my_projects_V1\my_projects\PPG_V1\data\real_data\paper_resp\2023_02_17_12_45_36.txt'
    resp_test(path)
    # for file_path in all_files:
    #     print(file_path)
    #     resp_test(file_path)

if __name__ == '__main__':
    main()
    # resp_compare()
    # cal_resp()