#coding:utf-86

import pywt
import matplotlib.pyplot as plt
from PyEMD import EMD, Visualisation
import numpy as np
from scipy.signal import hilbert
import pandas as pd
import glob
import scipy.signal as scisignal
import math
from scipy import signal
import openpyxl
from codes.process import VMD
import time


def read_data(path):
    new_data = []
    with open(path, 'r') as f:
        data_list = f.readlines()
        for i in range(len(data_list)):
            temp = data_list[i].strip().replace(" ", "")
            new_data.append(map(int, temp.split(",")))

    data = pd.DataFrame(new_data)
    data.columns = ["ir1", "red1", "ir2", "red2"]

    # plt.figure(figsize=(10, 86))
    # plt.subplot(211)
    # plt.plot(data.ir2, c="r")
    # plt.title("ir2")
    #
    # plt.subplot(212)
    # plt.plot(data.red2, c="b")
    # plt.title("red2")
    #
    # plt.subplots_adjust(hspace=0.86)
    # plt.show()

    return data

def timer(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        print("起始时间：", start)
        func(*args, **kwargs)
        end = time.time()
        print("结束时间为：", end)
        print("程序运行的时间为： ", end - start)
    return wrapper


def CalSQI(data):
    N = len(data)
    x_mean = np.mean(data)
    x_std = np.std(data)
    SQI = (1 / N) * sum([((x - x_mean) / x_std) ** 3 for x in data])
    return SQI

def signal_fft(data, freq):
    N = len(data)
    fs = freq
    df = fs / (N - 1)  # df分辨率
    # 构建频率数组
    f = [df * n for n in range(0, N)]
    Y = np.fft.fft(data) * 2 / N
    absY = [np.abs(x) for x in Y]

    # plt.plot(f, absY)
    # plt.show()
    return f, absY

def get_freq(freq_value, freq_energe):
    # 获取能量最大对应的频率
    N = int(len(freq_value) / 2)
    max_index = np.argmax(freq_energe[0:N])

    max_ap = np.max(freq_energe)
    freq = freq_value[max_index]

    return freq, max_ap


def get_dwt_res(data, w='db33', n=10):
    """Decompose and plot a signal S.
    S = An + Dn + Dn-1 + ... + D1
    """
    mode = pywt.Modes.smooth
    w = pywt.Wavelet(w)            # 选取小波函数
    a = data
    ca = []   # 近似分量, a表示低频近似部分
    cd = []   # 细节分量, b表示高频细节部分
    for i in range(n):
        (a, d) = pywt.dwt(a, w, mode)#进行n阶离散小波变换
        ca.append(a)
        cd.append(d)
    # print('---------------------')
    # print(len(ca), len(ca[0]), len(ca[1]), len(ca[2]), len(ca[3]), len(ca[4]), len(ca[5]), len(ca[6]), len(ca[7]), len(ca[86]), len(ca[9]))
    # print(type(ca))
    # print(ca)
    # print('+++++++++++++++++++++')
    # print(len(cd), len(cd[0]), len(cd[1]), len(cd[2]), len(cd[3]), len(cd[4]), len(cd[5]), len(cd[6]), len(cd[7]), len(cd[86]), len(cd[9]))
    #
    # print(type(cd))
    # print(cd)

    rec_a = []
    rec_d = []

    for i, coeff in enumerate(ca):
        coeff_list = [coeff, None] + [None] * i
        rec_a.append(pywt.waverec(coeff_list, w))      #重构

    for i, coeff in enumerate(cd):  # i, coeff 分别对应ca中的下标和元素，分了几层i就为几
        coeff_list = [None, coeff] + [None] * i
        rec_d.append(pywt.waverec(coeff_list, w))
    # print(rec_a, type(rec_a))
    # print(rec_d, type(rec_d))

    # fig = plt.figure(figsize=(16, 12))
    # ax_main = fig.add_subplot(len(rec_a) + 1, 1, 1)
    # # ax_main.set_title(title)
    # ax_main.plot(data)
    # ax_main.set_xlim(0, len(data) - 1)
    #
    # for i, y in enumerate(rec_a):
    #     ax = fig.add_subplot(len(rec_a) + 1, 2, 3 + i * 2)
    #     ax.plot(y, 'r')
    #     ax.set_xlim(0, len(y) - 1)
    #     ax.set_ylabel("A%d" % (i + 1))
    #     plt.subplots_adjust(hspace=0.5)
    #
    # for i, y in enumerate(rec_d):
    #     ax = fig.add_subplot(len(rec_d) + 1, 2, 4 + i * 2)
    #     ax.plot(y, 'g')
    #     ax.set_xlim(0, len(y) - 1)
    #     ax.set_ylabel("D%d" % (i + 1))
    #     plt.subplots_adjust(hspace=0.5)
    #
    # plt.show()

    return rec_a, rec_d

def emd_plot2(data):
    """EMD"""
    s = np.array(data)
    emd = EMD()
    emd.emd(s)
    imfs, res = emd.get_imfs_and_residue()
    # 绘图 IMF
    vis = Visualisation()
    vis.plot_imfs(imfs=imfs, residue=res, include_residue=True)
    # 绘制并显示所有提供的IMF的瞬时频率
    vis.plot_instant_freq(imfs=imfs)
    vis.show()
    return imfs

def emd_plot(data, freq=100):
    end = len(data) / freq
    # t = np.arange(0, end, 0.01)
    s = np.array(data)
    emd = EMD()
    emd.emd(s)
    imfs, res = emd.get_imfs_and_residue()
    # imfs = emd(data)
    # rint(imfs)
    print(len(imfs))  # 7
    # print(type(imfs))  # <class 'numpy.ndarray'>
    max_freq_list = list()
    plt.figure(figsize=(10, 8))
    for i in range(len(imfs)):
        plt.subplot(len(imfs), 1, i+1)
        # y_ppg, abs_y_ppg = signal_fft(imfs[i], freq)

        # freq, max_ap = get_freq(y_ppg, abs_y_ppg)
        # max_freq_list.append((freq, max_ap))
        plt.plot(imfs[i])
        plt.ylabel("D%d" % (i + 1))
        plt.subplots_adjust(hspace=0.5)
    plt.show()

    # # 绘图 IMF
    # vis = Visualisation()
    # vis.plot_imfs(imfs=imfs, residue=res, t=t, include_residue=True)
    # # 绘制并显示所有提供的IMF的瞬时频率
    # vis.plot_instant_freq(t, imfs=imfs)
    # vis.show()

    return imfs, max_freq_list

def bandpass_filter(data, fs=500, start_fs=0.1, end_fs=0.8):
    """巴特沃兹带通滤波"""
    winHz = [start_fs, end_fs]
    wn1 = 2 * winHz[0] / fs
    wn2 = 2 * winHz[1] / fs
    b, a = scisignal.butter(2, [wn1, wn2],
                            'bandpass')  # 截取频率[1，5]hz #这里假设采样频率为1000hz,信号本身最大的频率为500hz，要滤除10hz以下和400hz以上频率成分，即截至频率为10hz和400hz,则wn1=2*10/1000=0.02,wn2=2*400/1000=0.86。Wn=[0.02,0.86]#https://blog.csdn.net/weixin_37996604/article/details/82864680
    data = scisignal.filtfilt(b, a, data)  # data为要过滤的信号
    return data

def vmd(data):
    K = 8
    alpha = 5000
    tau = 1e-6
    vmd = VMD(K, alpha, tau)
    u, omega_K = vmd(data)
    results = data - u[0]
    return results, u, u[0]

def plot_vmd(data, u):
    plt.figure(figsize=(10, 8))
    plt.subplot(len(u) + 1, 1, 1)
    plt.plot(data)

    for i in range(len(u)):
        plt.subplot(len(u) + 1, 2, 3 + i * 2)
        plt.plot(u[i])
        plt.title('%s层分量' % i)
        plt.subplots_adjust(hspace=0.8)
        # plt.subplots_adjust(hspace=0.8)

    for i in range(len(u)):
        f, absY = signal_fft(u[i], 100)
        plt.subplot(len(u) + 1, 2, 4 + i * 2)
        plt.plot(f, absY)
        plt.title('%s层分量' % i)
        plt.subplots_adjust(hspace=0.8)

    plt.show()

@timer
def vmd_filter(data):
    filtered_data, u, first_component = vmd(data)

    print("raw_mean = ", np.mean(data))
    print("first_mean = ", np.mean(first_component))
    plt.figure(figsize=(10, 8))
    plt.subplot(211)
    plt.plot(data)
    plt.plot(first_component, c='r')
    plt.title("raw data")
    plt.subplot(212)
    plt.plot(filtered_data)
    plt.title("filtered data")
    plt.show()
    # plot_vmd(data, u)

    return filtered_data, first_component


def compare_test(data):
    ac, dc = get_dwt_res(data.ir2)

    ac_new = ac[8][:29000]
    ir2_new = data.ir2[:29000]

    vmded_data, first = vmd_filter(ir2_new)

    new_data = []
    for i in range(len(ac_new)):
        new_data.append(ir2_new[i] - ac_new[i])

    # print(len(ac_new))
    plt.plot(ir2_new)
    plt.plot(ac_new, c="r")
    plt.plot(new_data, c="g")
    plt.show()

    s_newdata = CalSQI(new_data[5000:6000])
    s_ir2new = CalSQI(ir2_new[5000:6000])
    s_vmd = CalSQI(vmded_data[5000:6000])
    print(s_ir2new, s_newdata, s_vmd)


def butter_test(data):
    filtered = bandpass_filter(data, start_fs=0.1, end_fs=5)

    s_data = CalSQI(data)
    s_buttered = CalSQI(filtered)

    vmded, first = vmd_filter(filtered)

    s_vmd = CalSQI(vmded)

    print(s_data, s_buttered, s_vmd)

    plt.figure(figsize=(10, 8))
    plt.subplot(311)
    plt.plot(data, c='r')
    plt.title("raw signal")
    plt.subplot(312)
    plt.plot(filtered, c='g')
    plt.plot(first, c='b')
    plt.title("filtered signal")
    plt.subplot(313)
    plt.plot(vmded)
    plt.title("vmded signal")
    plt.subplots_adjust(hspace=0.5)
    plt.show()

    return filtered

def vmd_test(data):
    filtered, u, first = vmd(data)
    remove_second = data - u[1]
    buttered = bandpass_filter(filtered, start_fs=0.1, end_fs=5)
    plt.figure(figsize=(10, 8))
    plt.subplot(311)
    plt.plot(data, c='r')
    plt.subplot(312)
    plt.plot(filtered, c='g')
    plt.subplot(313)
    plt.plot(buttered, c='b')
    plt.show()


def dwt_test(data):
    data = data.ir2.values.tolist()
    buttered = bandpass_filter(data, start_fs=0.1, end_fs=5)

    # ac, dc = get_dwt_res(data)
    ac, dc = get_dwt_res(buttered)
    resp_disturb = ac[-2][0:len(data)]
    # print(len(buttered), len(resp_disturb))

    # removed = [data[i] - resp_disturb[i] for i in range(len(data))]
    removed = [buttered[i] - resp_disturb[i] for i in range(len(data))]
    # buttered = bandpass_filter(removed, start_fs=0.1, end_fs=5)

    plt.subplot(311)
    plt.plot(data)
    plt.subplot(312)
    # plt.plot(data)
    plt.plot(buttered)
    plt.plot(resp_disturb, c='r')
    plt.subplot(313)
    plt.plot(removed)
    plt.show()



if __name__ == '__main__':

    file_path = r"E:\my_projects\PPG\data\3disturb_5pulse\100\20220918201446_100.txt"
    data = read_data(file_path)

    # butter_test(data.ir2)
    # compare_test(data)
    # vmd_test(data.ir2)
    # emd_plot(data.ir2[5000:9000], freq=500)

    dwt_test(data)
    print("-------------")
    # vmd_filter(data.ir2)



