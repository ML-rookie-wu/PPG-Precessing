# -*- coding: utf-8 -*-

"""
@author: Mark Wu
@file: CEEMDAN_Test.py
@time: 2023/2/9 17:03
"""
import matplotlib.pyplot as plt
import numpy as np
from PyEMD import CEEMDAN
from codes.utils.GetFileData import travel_dir, get_ir2
from codes.utils.GetFFT import signal_fft, get_freq
from codes.utils.MyFilters import bandpass_filter


def ceemdan_decompose_res(data):
    data = np.array(data)
    ceemdan = CEEMDAN()
    ceemdan.ceemdan(data)
    imfs, res = ceemdan.get_imfs_and_residue()
    print(len(imfs))
    IImfs = []
    plt.figure(figsize=(12, 9))
    plt.subplots_adjust(hspace=0.1)
    plt.subplot(imfs.shape[0]+3, 1, 1)
    plt.plot(data, 'r')
    for i in range(imfs.shape[0]):
        plt.subplot(imfs.shape[0]+3, 1, i+2)
        plt.plot(imfs[i], 'g')
        plt.ylabel("IMF %i" %(i+1))
        plt.locator_params(axis='x', nbins=10)
        # 在函数前必须设置一个全局变量 IImfs=[]
        IImfs.append(imfs[i])
    plt.subplot(imfs.shape[0]+3, 1, imfs.shape[0]+3)
    plt.plot(res, 'g')
    plt.show()

def ceemdan(data):
    data = np.array(data)
    ceemdan = CEEMDAN(trials=500, parallel=True, processes=4)
    ceemdan.ceemdan(data)
    imfs, res = ceemdan.get_imfs_and_residue()
    fig = plt.figure(figsize=(16, 12))
    ax_main = fig.add_subplot(len(imfs) + 1, 1, 1)
    # ax_main.set_title(title)
    ax_main.plot(data)
    ax_main.set_xlim(0, len(data) - 1)

    for i, y in enumerate(imfs):
        ax = fig.add_subplot(len(imfs) + 1, 2, 3 + i * 2)
        ax.plot(y, 'r')
        ax.set_xlim(0, len(y) - 1)
        ax.set_ylabel("A%d" % (i + 1))
        plt.subplots_adjust(hspace=0.5)

    for i, y in enumerate(imfs):
        ax = fig.add_subplot(len(imfs) + 1, 2, 4 + i * 2)
        f, abs = signal_fft(data, freq=500)
        ax.plot(f, abs, color="deeppink")
        # ax.set_xlim(0, len(y) - 1)
        ax.set_ylabel("F%d" % (i + 1))
        plt.subplots_adjust(hspace=0.5)

    plt.show()

    plt.figure(figsize=(10, 8))
    restruct = imfs[0]
    for i in range(1, len(imfs)):
        restruct += imfs[i]
    plt.subplot(211)
    plt.plot(data)
    plt.subplot(212)
    plt.plot(np.array(data) - res)
    plt.show()


def main():
    path = r'D:\my_projects_V1\my_projects\PPG_V1\data\real_data\paper_resp\2023_02_17_12_06_36.txt'
    # path = r'D:\my_projects_V1\my_projects\PPG_V1\data\spo2_compare_new\1disturb_3pulse\98\20221129195154_98.txt'
    ir2 = get_ir2(path)
    butter = bandpass_filter(ir2, start_fs=0.1, end_fs=5)
    # ceemdan_decompose_res(ir2[0:8000])
    ceemdan(butter)


if __name__ == '__main__':
    main()

