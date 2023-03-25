# -*- coding: utf-86 -*-
# @Time : 2022/10/13 20:42
# @Author : Mark Wu
# @FileName: ppg_quality.py
# @Software: PyCharm

'''
########################################################################

灌注指数：P_SQI = ((y_max - y_min) / abs(x_mean)) * 100

y_max、y_min表示的是滤波后信号的最大值、最小值，x_mean表示原始信号的均值
The perfusion index is the ratio of the pulsatile blood flow to the nonpulsatile
or static blood in peripheral tissue
########################################################################

########################################################################

偏斜度：S_SQI = (1 / len(data)) * sum([(x_i - x_mean / std) ** 3 for x_i in data])

x_mean表示均值，std表示标准差
skewness is associated with corrupted PPG signals

########################################################################

########################################################################

峰度：K_SQI = (1 / len(data)) * sum([(x_i - x_mean / std) ** 4 for x_i in data])

Kurtosis is a statistical measure used to describe the distribution of observed
data around the mean
########################################################################

########################################################################

熵：E_SQI = -(sum([(x_i ** 2) * np.log(x_i ** 2) for x_i in data])

Entropy quantifies how much the probability density function (PDF) of the signal
differs from a uniform distribution and thus provides a quantitative measure of the uncertainty
present in the signal
########################################################################

########################################################################

零交叉率：Z_SQI = (1 / len(data)) * sum([1 if x_i > 0 else 0 for x_i in data])

This is the rate of sign-changes in the processed signal, that is, the rate
at which the signal changes from positive to negative or back
data为滤波后的数据
########################################################################

########################################################################

信噪比：N_SQI = std_filtered / std_raw

std_filtered表示滤波后的信号的标准差，std_raw表示原始信号的标准差
This is a measure used in science and engineering that compares
the level of a desired signal to the level of background noise. There are many ways to define
signal-to-noise ratio
########################################################################
'''

import numpy as np


def P_SQI(raw_data, filtered_data):
    y_max = max(filtered_data)
    y_min = min(filtered_data)
    x_mean = np.mean(raw_data)
    SQI = ((y_max - y_min) / abs(x_mean)) * 100

    return SQI

def SNR(raw_data, filtered_data):
    p_raw = (1 / len(raw_data)) * sum(x ** 2 for x in raw_data)
    p_filtered = (1 / len(filtered_data)) * sum(y ** 2 for y in filtered_data)
    noisy = np.array(p_raw) - np.array(p_filtered)



a = [1, 2, 3]
b = [4, 5, 6]
print(np.array(a) - np.array(b))

