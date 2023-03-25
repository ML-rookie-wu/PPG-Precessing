#coding:utf-8
import numpy as np
import matplotlib.pyplot as plt
from PyEMD import EEMD, EMD, CEEMDAN
from codes.utils.GetFileData import read_from_file
from codes.process import get_dwt_res, bandpass_filter
from codes.utils.GetFFT import get_freq, signal_fft

def Signal():
    global E_imfNo
    E_imfNo = np.zeros(50, dtype=np.int)

    # EEMD options
    max_imf = -1

    """
    信号参数：
    N:采样频率500Hz
    tMin:采样开始时间
    tMax:采样结束时间 2*np.pi
    """
    N = 500
    tMin, tMax = 0, 2 * np.pi
    T = np.linspace(tMin, tMax, N)
    # 信号S:是多个信号叠加信号
    S = 3 * np.sin(4 * T) + 4 * np.cos(9 * T) + np.sin(8.11 * T + 1.2)

    # EEMD计算
    eemd = EEMD()
    eemd.trials = 50
    eemd.noise_seed(12345)

    E_IMFs = eemd.eemd(S, T, max_imf)
    imfNo = E_IMFs.shape[0]


    # Plot results in a grid
    c = np.floor(np.sqrt(imfNo + 1))
    r = np.ceil((imfNo + 1) / c)

    plt.ioff()
    plt.subplot(r, c, 1)
    plt.plot(T, S, 'r')
    plt.xlim((tMin, tMax))
    plt.title("Original signal")

    for num in range(imfNo):
        plt.subplot(r, c, num + 2)
        plt.plot(T, E_IMFs[num], 'g')
        plt.xlim((tMin, tMax))
        plt.title("Imf " + str(num + 1))

    plt.show()

def emd_plot(data, freq=500):
    end = len(data) / freq
    # t = np.arange(0, end, 0.01)
    s = np.array(data)
    emd = EMD()
    emd.emd(s)
    imfs, res = emd.get_imfs_and_residue()
    # imfs = emd(data)
    # rint(imfs)
    print(len(imfs))  # 7

    max_freq_list = list()
    plt.figure(figsize=(10, 8))
    for i in range(len(imfs)):
        plt.subplot(len(imfs), 1, i+1)
        plt.plot(imfs[i])
        plt.ylabel("D%d" % (i + 1))
        plt.subplots_adjust(hspace=0.5)
    plt.show()

    return imfs, max_freq_list

def eemd(data):
    # EEMD计算
    eemd = EEMD(trials=50, parallel=True, processes=4)
    eemd.noise_seed(12345)
    E_IMFs = eemd.eemd(data)
    print(len(E_IMFs))
    restruct_list = []
    fig = plt.figure(figsize=(16, 12))
    ax_main = fig.add_subplot(len(E_IMFs) + 1, 1, 1)
    ax_main.plot(data)
    ax_main.set_xlim(0, len(data) - 1)

    for i, x in enumerate(E_IMFs):
        ax = fig.add_subplot(len(E_IMFs) + 1, 2, 3 + i*2)
        ax.plot(x, 'r')
        ax.set_xlim(0, len(x) - 1)
        ax.set_ylabel("A%d" % (i + 1))
        plt.subplots_adjust(hspace=0.8)

    for i, y in enumerate(E_IMFs):
        ax = fig.add_subplot(len(E_IMFs) + 1, 2, 4 + i * 2)
        f, absY = signal_fft(y)
        freq, max_amp = get_freq(f, absY)
        if 0.1 < freq < 5:
            restruct_list.append(E_IMFs[i])
        ax.plot(f, absY, 'g')
        # ax.set_xlim(0, len(y) - 1)
        ax.set_ylabel("D%d" % (i + 1))
        plt.subplots_adjust(hspace=0.8)

    plt.show()

    restruct_data = np.array(restruct_list[0])
    print("len-----", len(restruct_list))
    if len(restruct_list) > 1:
        for i in range(1, len(restruct_list)):
            restruct_data += np.array(restruct_list[i])
    plt.figure(figsize=(10, 8))
    plt.subplot(311)
    plt.plot(data)
    plt.subplot(312)
    plt.plot(restruct_data)
    plt.subplot(313)
    plt.plot(bandpass_filter(data, start_fs=0.1, end_fs=5))
    plt.show()
    return E_IMFs

def compare():
    path = r'D:\my_projects_V1\my_projects\pyqt\learnDemo\data\2022_5_3_15_48.txt'
    # path = r'D:\my_projects_V1\my_projects\PPG_V1\data\spo2_compare_test\1disturb_3pulse\100\20221129193845_100.txt'
    data = read_from_file(path)
    red2 = np.array(data.red2)
    # fil_red2 = bandpass_filter(red2, start_fs=0.1, end_fs=5)
    print(type(red2))
    # eemd(fil_red2[0:4000])
    eemd(red2)
    # emd_plot(red2)
    # get_dwt_res(red2)




if __name__ == '__main__':
    compare()