# -*- coding: utf-8 -*-

"""
@author: Mark Wu
@file: BRAnalysis.py
@time: 2023/3/2 9:30
"""

import pandas as pd
import matplotlib.pyplot as plt
from codes.utils.MyFilters import vmd, bandpass_filter
from codes.utils.GetFFT import signal_fft, get_freq

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


def get_csv(path):
    data = pd.read_csv(path, sep=",")
    return data

def test1():
    path = r"D:\my_projects_V1\my_projects\PPG_V1\data\BR\apnea_csv\wu\20230302135937.csv"
    record = get_csv(path)
    ir = record.ir
    ir = ir[0:6000].reset_index(drop=True)
    resp = record.resp
    resp = resp[0:6000].reset_index(drop=True)
    spo2 = record.spo2
    pr = record.pr

    ir_butter = bandpass_filter(ir, fs=100, start_fs=0.1, end_fs=5)
    f, absY = signal_fft(ir_butter, freq=100)
    plt.figure(figsize=(10, 8))
    plt.subplot(211)
    plt.plot(ir_butter)
    plt.subplot(212)
    plt.plot(f, absY)
    plt.show()

    results, imfs, vmd_resp = vmd(ir)
    fig = plt.figure(figsize=(16, 12))
    ax_main = fig.add_subplot(len(imfs) + 2, 1, 1)
    ax_main.plot(ir, color="darkviolet")
    ax_main.set_title("原始PPG信号")
    ax_main.set_ylabel("幅值")
    ax_main.set_xlabel("采样点")
    ax_main.set_xlim(0, len(ir) - 1)

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

    butter_resp = bandpass_filter(vmd_resp, fs=100, start_fs=0.2, end_fs=0.7)
    resp_freq, resp_amp = signal_fft(butter_resp, freq=100)
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
    plt.show()


def test2():
    path = r"D:\my_projects_V1\my_projects\PPG_V1\data\BR\apnea_csv\wu\20230302135937.csv"
    record = get_csv(path)
    ir = record.ir
    # ir = ir[0:6000].reset_index(drop=True)
    resp = record.resp
    # resp = resp[0:6000].reset_index(drop=True)
    spo2 = record.spo2
    pr = record.pr
    rrInterval = record.rrInterval

    plt.figure(figsize=(10, 8))
    plt.subplot(511)
    plt.plot(ir)
    plt.subplot(512)
    plt.plot(resp)
    plt.subplot(513)
    plt.plot(spo2)
    plt.subplot(514)
    plt.plot(pr)
    plt.subplot(515)
    plt.plot(rrInterval)
    plt.show()

if __name__ == '__main__':
    # test1()
    test2()