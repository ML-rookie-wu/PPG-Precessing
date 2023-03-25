# -*- coding: utf-8 -*-

"""
@author: Mark Wu
@file: WaveletTest1.py
@time: 2023/2/15 14:24
"""
import matplotlib.pyplot as plt
import pywt
from codes.utils.GetFileData import read_from_file


def get_data(path):
    data = read_from_file(path)
    return data

# Get data:
ecg = pywt.data.ecg()  # 生成心电信号
index = []
# data = []
# for i in range(len(ecg)-1):
#     X = float(i)
#     Y = float(ecg[i])
#     index.append(X)
#     data.append(Y)

path = r'D:\my_projects_V1\my_projects\PPG_V1\data\spo2_compare_new\3disturb_3pulse\98\20221130141715_98.txt'
data = get_data(path)
data = data.ir2

# Create wavelet object and define parameters
w = pywt.Wavelet('db8')  # 选用Daubechies8小波
maxlev = pywt.dwt_max_level(len(data), w.dec_len)
print("maximum level is " + str(maxlev))
threshold = 1.12 # Threshold for filtering

# Decompose into wavelet components, to the level selected:
coeffs = pywt.wavedec(data, 'db8', level=maxlev)  # 将信号进行小波分解

# plt.figure()
for i in range(1, len(coeffs)):
    coeffs[i] = pywt.threshold(coeffs[i], threshold*max(coeffs[i]))  # 将噪声滤波

datarec = pywt.waverec(coeffs, 'db8')  # 将信号进行小波重构

mintime = 0
maxtime = mintime + len(data) + 1

plt.figure()
plt.subplot(2, 1, 1)
plt.plot(data[mintime:maxtime])
plt.xlabel('time (s)')
plt.ylabel('microvolts (uV)')
plt.title("Raw signal")
plt.subplot(2, 1, 2)
plt.plot(datarec[mintime:maxtime-1])
plt.xlabel('time (s)')
plt.ylabel('microvolts (uV)')
plt.title("De-noised signal using wavelet techniques")

plt.tight_layout()
plt.show()
