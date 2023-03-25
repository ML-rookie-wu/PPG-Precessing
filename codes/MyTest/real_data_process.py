#coding:utf-8
from codes.utils.GetFileData import read_from_file
from codes.utils.MyFilters import bandpass_filter, get_dwt_res
from codes.utils.GetFFT import signal_fft, get_freq
# from codes.process import get_dwt_res
import matplotlib.pyplot as plt
from codes.MyTest.eemd_test import eemd
import numpy as np
import pywt


def get_data(path):
    raw_data = read_from_file(path)
    # data = raw_data.ir2
    return raw_data

def test1(data):
    buttered = bandpass_filter(data, start_fs=0.1, end_fs=5)
    ac, dc = get_dwt_res(buttered)
    resp_disturb = ac[-2][0:len(data)]
    removed = [buttered[i] - resp_disturb[i] for i in range(len(data))]
    plt.figure(figsize=(10, 8))
    plt.subplot(311)
    plt.plot(data)
    plt.title("raw data")
    plt.subplot(312)
    plt.plot(buttered)
    plt.title("buttered")
    plt.subplot(313)
    plt.plot(removed)
    plt.subplots_adjust(hspace=0.8)
    plt.show()

def test2(data):
    """
    eemd测试
    :param data:
    :return:
    """
    buttered_data = bandpass_filter(data, start_fs=0.1, end_fs=5)
    data = np.array(data)
    IMFs = eemd(data)
    num = len(IMFs)
    fig1 = plt.figure(figsize=(10, 8))
    ax_raw = fig1.add_subplot(num+1, 1, 1)
    ax_raw.set_title("raw data")
    ax_raw.plot(data)
    for index, imf in enumerate(IMFs):
        ax = fig1.add_subplot(num+1, 1, index+2)
        ax.set_title("imf_%s" % (index+1))
        ax.plot(IMFs[index])
        plt.subplots_adjust(hspace=0.8)
    plt.show()

    fig2 = plt.figure(figsize=(10, 8))
    freq_ax_raw = fig2.add_subplot(num+1, 1, 1)
    freq_ax_raw.set_title("raw data frequency")
    f_raw, absY_raw = signal_fft(buttered_data)
    freq_ax_raw.plot(f_raw, absY_raw)
    for i, y in enumerate(IMFs):
        ax = fig2.add_subplot(len(IMFs) + 1, 1, i+2)
        f, absY = signal_fft(y)
        ax.plot(f, absY)
        # ax.set_xlim(0, len(y) - 1)
        ax.set_title("imf%s frequency" % (i + 1))
        plt.subplots_adjust(hspace=0.8)
    plt.show()

    res = IMFs[-5] + IMFs[-4]
    f_res, absY_res = signal_fft(res)
    plt.figure(figsize=(10, 8))
    plt.subplot(211)
    plt.plot(res)
    plt.subplot(212)
    plt.plot(f_res, absY_res)
    plt.show()

def test3(data):
    """
    小波变换测试
    :param data:
    :return:
    """
    buttered = bandpass_filter(data, start_fs=0.1, end_fs=5)
    ac, dc = get_dwt_res(buttered)
    num = len(ac)
    fig1 = plt.figure(figsize=(10, 8))
    ax_raw = fig1.add_subplot(num + 1, 1, 1)
    ax_raw.set_title("raw data")
    ax_raw.plot(buttered)
    for index, imf in enumerate(ac):
        ax = fig1.add_subplot(num + 1, 1, index + 2)
        ax.set_title("imf_%s" % (index + 1))
        ax.plot(imf)
        plt.subplots_adjust(hspace=1)
        # ax.tight_layout()
        # plt.tight_layout()
    plt.show()

    fig2 = plt.figure(figsize=(10, 8))
    freq_ax_raw = fig2.add_subplot(num + 1, 1, 1)
    freq_ax_raw.set_title("raw data frequency")
    f_raw, absY_raw = signal_fft(buttered)
    freq_ax_raw.plot(f_raw, absY_raw)
    for i, y in enumerate(ac):
        ax = fig2.add_subplot(len(ac) + 1, 1, i + 2)
        f, absY = signal_fft(y)
        ax.plot(f, absY)
        # ax.set_xlim(0, len(y) - 1)
        ax.set_title("imf%s frequency" % (i + 1))
        plt.subplots_adjust(hspace=0.8)
    plt.show()

def test4(data):
    y = data
    N = len(data)

    fs = 500
    df = fs / (N - 1)  # df分辨率
    # 构建频率数组
    f = [df * n for n in range(0, N)]
    frq1 = f

    wavename = 'db5'
    cA, cD = pywt.dwt(y, wavename)
    print(len(cA), len(cD))
    ya = pywt.idwt(cA, None, wavename, 'smooth')  # approximated component
    yd = pywt.idwt(None, cD, wavename, 'smooth')  # detailed component
    x = range(len(y))
    plt.figure(figsize=(12, 9))
    plt.subplot(311)
    plt.plot(y)
    plt.title('original signal')
    plt.subplot(312)
    plt.plot(ya)
    plt.title('approximated component')
    plt.subplot(313)
    plt.plot(yd)
    plt.title('detailed component')
    plt.tight_layout()
    plt.show()

    # 图像单边谱
    plt.figure(figsize=(12, 9))
    plt.subplot(311)
    data_f = abs(np.fft.fft(cA)) / N
    data_f1 = data_f[range(int(N / 2))]
    plt.plot(frq1, data_f, 'red')

    plt.subplot(312)
    data_ff = abs(np.fft.fft(cD)) / N
    data_f2 = data_ff[range(int(N / 2))]
    plt.plot(frq1, data_ff, 'k')

    plt.xlabel('pinlv(hz)')
    plt.ylabel('amplitude')

    plt.show()

def test_resp_eemd(data):
    resp = data.resp
    ir2 = data.ir2
    red2 = data.red2
    red2_resp = bandpass_filter(red2, start_fs=0.1, end_fs=0.6)
    buttered_resp = bandpass_filter(resp, start_fs=0.1, end_fs=0.8)
    buttered_data = bandpass_filter(red2, start_fs=0.1, end_fs=5)
    data = np.array(red2)
    IMFs = eemd(data)
    num = len(IMFs)
    fig1 = plt.figure(figsize=(10, 8))
    ax_raw = fig1.add_subplot(num + 1, 1, 1)
    ax_raw.set_title("raw data")
    ax_raw.plot(data)
    for index, imf in enumerate(IMFs):
        ax = fig1.add_subplot(num + 1, 1, index + 2)
        ax.set_title("imf_%s" % (index + 1))
        ax.plot(IMFs[index])
        plt.subplots_adjust(hspace=0.8)
    plt.show()

    fig2 = plt.figure(figsize=(10, 8))
    freq_ax_raw = fig2.add_subplot(num + 1, 1, 1)
    freq_ax_raw.set_title("raw data frequency")
    f_raw, absY_raw = signal_fft(buttered_data)
    freq_ax_raw.plot(f_raw, absY_raw)
    for i, y in enumerate(IMFs):
        ax = fig2.add_subplot(len(IMFs) + 1, 1, i + 2)
        f, absY = signal_fft(y)
        ax.plot(f, absY)
        # ax.set_xlim(0, len(y) - 1)
        ax.set_title("imf%s frequency" % (i + 1))
        plt.subplots_adjust(hspace=0.8)
    plt.show()

    res = IMFs[-3] + IMFs[-4]
    res = bandpass_filter(res, start_fs=0.1, end_fs=0.6)
    f_res, absY_res = signal_fft(res)
    plt.figure(figsize=(10, 8))
    plt.subplot(411)
    plt.plot(res)
    plt.subplot(412)
    plt.plot(f_res, absY_res)
    plt.subplot(413)
    plt.plot(buttered_resp)
    plt.subplot(414)
    plt.plot(red2_resp)
    plt.subplots_adjust(hspace=0.8)
    plt.show()

def test_resp_eemd(data):
    resp = data.resp
    ir2 = data.ir2
    red2 = data.red2
    red2_resp = bandpass_filter(red2, start_fs=0.1, end_fs=0.6)
    buttered_resp = bandpass_filter(resp, start_fs=0.1, end_fs=0.8)
    buttered_data = bandpass_filter(red2, start_fs=0.1, end_fs=5)
    data = np.array(red2)
    IMFs = eemd(data)
    num = len(IMFs)
    fig1 = plt.figure(figsize=(10, 8))
    ax_raw = fig1.add_subplot(num + 1, 1, 1)
    ax_raw.set_title("raw data")
    ax_raw.plot(data)
    for index, imf in enumerate(IMFs):
        ax = fig1.add_subplot(num + 1, 1, index + 2)
        ax.set_title("imf_%s" % (index + 1))
        ax.plot(IMFs[index])
        plt.subplots_adjust(hspace=0.8)
    plt.show()

    fig2 = plt.figure(figsize=(10, 8))
    freq_ax_raw = fig2.add_subplot(num + 1, 1, 1)
    freq_ax_raw.set_title("raw data frequency")
    f_raw, absY_raw = signal_fft(buttered_data)
    freq_ax_raw.plot(f_raw, absY_raw)
    for i, y in enumerate(IMFs):
        ax = fig2.add_subplot(len(IMFs) + 1, 1, i + 2)
        f, absY = signal_fft(y)
        ax.plot(f, absY)
        # ax.set_xlim(0, len(y) - 1)
        ax.set_title("imf%s frequency" % (i + 1))
        plt.subplots_adjust(hspace=0.8)
    plt.show()

    res = IMFs[-3] + IMFs[-4]
    res = bandpass_filter(res, start_fs=0.1, end_fs=0.6)
    f_res, absY_res = signal_fft(res)
    f_resp, absY_resp = signal_fft(red2_resp)
    plt.figure(figsize=(10, 8))
    plt.subplot(511)
    plt.plot(res)
    plt.subplot(512)
    plt.plot(f_res, absY_res)
    plt.subplot(513)
    plt.plot(buttered_resp)
    plt.subplot(514)
    plt.plot(red2_resp)
    plt.subplot(515)
    plt.plot(f_resp, absY_resp)
    plt.subplots_adjust(hspace=0.8)
    plt.show()


if __name__ == '__main__':
    # path = r'E:\my_projects\PPG\data\real_data\wu\20221205103917.txt'
    # path = r'E:\my_projects\PPG\data\spo2_compare_new\2disturb_3pulse\100\20221130134838_100.txt'
    path = r'E:\my_projects\PPG\data\real_data\wu\resp\2022_12_08_16_31_24.txt'
    data = get_data(path)

    # test1(data)
    # test2(data)
    # test3(data)
    # test4(data)
    test_resp_eemd(data)
