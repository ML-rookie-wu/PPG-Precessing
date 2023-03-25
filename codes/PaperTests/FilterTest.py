# -*- coding: utf-8 -*-

"""
@author: Mark Wu
@file: FilterTest.py
@time: 2023/2/7 16:43
"""

from scipy import signal
import matplotlib.pyplot as plt
from codes.utils.GetFileData import read_from_file
from codes.utils.MyFilters import bandpass_filter
from codes.utils.GetFFT import get_freq, signal_fft

def get_data(path):
    data = read_from_file(path)
    ir2 = data.ir2
    return ir2


def apply_chebyshev_filter(data, fs, ftype, freqs=[], order=2, rp=0.002):
    nyq = 0.5 * fs

    if ftype == 'low_pass':
        assert len(freqs) == 1
        cut = freqs[0] / nyq
        b, a = signal.cheby1(order, rp, cut, btype='lowpass')
    elif ftype == 'high_pass':
        assert len(freqs) == 1
        cut = freqs[0] / nyq
        b, a = signal.cheby1(order, rp, cut, btype='highpass')
    elif ftype == 'band_pass':
        assert len(freqs) == 2
        lowcut, highcut = freqs[0] / nyq, freqs[1] / nyq
        b, a = signal.cheby2(order, rp, [lowcut, highcut], btype='bandpass')
    elif ftype == 'band_stop':
        assert len(freqs) == 2
        lowcut, highcut = freqs[0] / nyq, freqs[1] / nyq
        b, a = signal.cheby1(order, rp, [lowcut, highcut], btype='bandstop')

    filtered = signal.lfilter(b, a, data)
    return filtered


def kalman_filter(data, q=0.0001, r=0.01):
    # 后验初始值
    x0 = data[0]                              # 令第一个估计值，为当前值
    p0 = 1.0
    # 存结果的列表
    x = [x0]
    for z in data[1:]:                        # kalman 滤波实时计算，只要知道当前值z就能计算出估计值(后验值)x0
        # 先验值
        x1_minus = x0                         # X(k|k-1) = AX(k-1|k-1) + BU(k) + W(k), A=1,BU(k) = 0
        p1_minus = p0 + q                     # P(k|k-1) = AP(k-1|k-1)A' + Q(k), A=1
        # 更新K和后验值
        k1 = p1_minus / (p1_minus + r)        # Kg(k)=P(k|k-1)H'/[HP(k|k-1)H' + R], H=1
        x0 = x1_minus + k1 * (z - x1_minus)   # X(k|k) = X(k|k-1) + Kg(k)[Z(k) - HX(k|k-1)], H=1
        p0 = (1 - k1) * p1_minus              # P(k|k) = (1 - Kg(k)H)P(k|k-1), H=1
        x.append(x0)                          # 由输入的当前值z 得到估计值x0存入列表中，并开始循环到下一个值
    return x

def savgol(data):
    filtered = signal.savgol_filter(data, 999, 3, mode="nearest")
    return filtered


def filter_plot():
    # path = r'D:\my_projects_V1\my_projects\PPG_V1\data\spo2_compare_new\3disturb_3pulse\98\20221130141715_98.txt'
    path = r'D:\my_projects_V1\my_projects\pyqt\learnDemo\data\2022_5_3_15_48.txt'
    data = get_data(path)
    # filtered_data = apply_chebyshev_filter(data, fs=500, freqs=[0.1, 5], ftype="band_pass")
    butter = bandpass_filter(data, start_fs=0.1, end_fs=5)
    kalman = kalman_filter(data)
    savgoled = savgol(data)

    raw_freq, raw_absY = signal_fft(data, freq=500)
    kalman_freq, kalman_absY = signal_fft(kalman, freq=500)
    butter_freq, butter_absY = signal_fft(butter, freq=500)

    plt.figure(figsize=(10, 8))
    plt.subplot(611)
    plt.plot(data)
    plt.subplot(612)
    plt.plot(kalman)
    plt.subplot(613)
    plt.plot(butter)
    plt.subplot(614)
    plt.plot(savgoled)
    plt.subplot(615)
    plt.plot(kalman_freq, kalman_absY)
    plt.subplot(616)
    plt.plot(butter_freq, butter_absY)

    plt.show()


def main():
    filter_plot()

if __name__ == '__main__':
    main()