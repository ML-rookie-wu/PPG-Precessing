#coding:utf-8
import numpy as np


def signal_fft(data, freq=500):
    N = len(data)
    fs = freq
    df = fs / (N - 1)  # df分辨率
    # 构建频率数组
    f = [df * n for n in range(0, N)]
    Y = np.fft.fft(data) * 2 / N
    absY = [np.abs(x) for x in Y]

    return f, absY


def my_fft(data, freq=500, num=1):
    N = len(data)
    fs = freq
    df = fs / (N*num - 1)  # df分辨率
    # 构建频率数组
    f = [df * n for n in range(0, N)]
    Y = np.fft.fft(data) * 2 / N
    absY = [np.abs(x) for x in Y]

    return f, absY


def get_freq(freq_value, freq_energe):
    """
    获取频率最大值及其对应的能量
    :param freq_value: 频率序列
    :param freq_energe: 频率对应的幅值序列
    :return:
    """
    # 获取能量最大对应的频率
    N = int(len(freq_value) / 2)
    max_index = np.argmax(freq_energe[0:N])

    max_index2 = np.argsort(freq_energe[0:N])[-2]
    max_ap = np.max(freq_energe)

    freq = freq_value[max_index]
    freq2 = freq_value[max_index2]

    return freq, max_ap