# -*- coding: utf-8 -*-

"""
@author: Mark Wu
@file: peakTest.py
@time: 2023/1/11 15:41
"""

import numpy as np
import pandas as pd
import scipy
import os
import matplotlib.pyplot as plt
from codes.utils.GetFileData import read_from_file
from codes.utils.MyFilters import reverse, bandpass_filter, kalman_filter
from codes.utils.Normal import normalization
from codes.PaperTests import PAPER_FIGURE_PATH

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

def read_data(path, label="simulator"):
    data = read_from_file(path)
    ir2 = data.ir2
    red2 = data.red2

    if label == "simulator":
        ir2 = list(reversed(ir2))
        red2 = list(reversed(red2))
    elif label == "real":
        ir2 = reverse(ir2)
        red2 = reverse(red2)
    return ir2, red2


def peaks(path):
    # path = r'D:\my_projects_V1\my_projects\PPG_V1\data\spo2_compare_new\0disturb_3pulse\98\20221130184623_98.txt'
    # path = r'D:\my_projects_V1\my_projects\PPG_V1\data\real_data\wu\20221205103726.txt'
    # path = r'D:\my_projects_V1\my_projects\pyqt\learnDemo\data\2022_5_3_15_48.txt'

    # data = read_from_file(path)
    # ir2 = data.ir2
    # rev_ir2 = list(reversed(ir2))
    # # rev_ir2 = reverse(ir2)
    # red2 = data.red2
    # rev_red2 = reverse(red2)

    ir2, red2 = read_data(path, label="real")
    peaks = []
    # but_ir2 = bandpass_filter(rev_ir2, start_fs=0.1, end_fs=3)
    but_ir2 = bandpass_filter(ir2, start_fs=0.1, end_fs=5)
    ir2_max = np.max(but_ir2)
    ir2_min = np.min(but_ir2)
    ir2_mean = (sum(but_ir2) - ir2_max - ir2_min) / (len(but_ir2) - 2)

    diff = np.diff(but_ir2)
    diff_max = np.max(diff)
    diff_min = np.min(diff)
    diff_mean = (sum(diff) - diff_max - diff_min) / (len(diff) - 2)
    th1 = 1.2 * diff_mean

    normal_ir2 = normalization(but_ir2)
    normal_diff = normalization(diff)
    for i in range(len(diff)-2):
        if diff[i] > 0 and diff[i+1] < 0:
            index = i+1
            for j in range(0, 4):
                if but_ir2[index+j] - but_ir2[index+j-1] > 0 and but_ir2[index+j+1] - but_ir2[index+j] < 0:
                    peak_index = index+j
                    if len(peaks) == 0:
                        peaks.append(peak_index)
                    else:
                        if peak_index - peaks[-1] < 300:
                            break
                        else:
                            peaks.append(peak_index)
    print(peaks)
    peaks_y = [normal_ir2[x] for x in peaks]

    plt.figure(figsize=(10, 8))
    # plt.subplot(211)
    # plt.plot(ir2)
    # plt.title("ir2")
    # plt.subplot(212)
    # plt.plot(but_ir2)
    plt.plot(normal_ir2, label='PPG')
    # plt.plot(normal_diff, c="r", label="diff")
    # plt.plot(normal_ir2, label='ppg')
    plt.ylabel("幅值")
    plt.xlabel("采样点")
    # plt.plot(normal_diff, c="r", label="diff")
    # plt.axhline(0, color='black', label="y=0")

    plt.scatter(peaks, peaks_y, marker='*', color="red", label="peak")

    # plt.axhline(th1, color='purple')
    # plt.axhline(ir2_mean, color="green")
    plt.legend(loc="best")
    plt.title("动态差分阈值法")
    plt.savefig(os.path.join(PAPER_FIGURE_PATH, "动态差分阈值法"), dpi=300)
    plt.show()

def peakTest(filtered_data):
    peakList = []
    ir2_max = np.max(filtered_data)
    ir2_min = np.min(filtered_data)
    ir2_mean = (sum(filtered_data) - ir2_max - ir2_min) / (len(filtered_data) - 2)

    diff = np.diff(filtered_data)
    diff_max = np.max(diff)
    diff_min = np.min(diff)
    diff_mean = (sum(diff) - diff_max - diff_min) / (len(diff) - 2)

    th1 = 1.2 * diff_mean
    normal_ir2 = normalization(filtered_data)
    normal_diff = normalization(diff)
    for i in range(len(diff) - 2):
        if diff[i] > 0 and diff[i + 1] < 0:
            index = i + 1
            for j in range(0, 4):
                if index+j < len(filtered_data) and filtered_data[index + j] - filtered_data[index + j - 1] > 0 and filtered_data[index + j + 1] - filtered_data[index + j] < 0:
                    peak_index = index + j
                    # print(peak_index)
                    if len(peakList) == 0:
                        peakList.append(peak_index)
                    else:
                        if peak_index - peakList[-1] < 200:
                            break
                        else:
                            peakList.append(peak_index)
    return peakList

def AMPD(data):
    """
    实现AMPD算法
    :param data: 1-D numpy.ndarray
    :return: 波峰所在索引值的列表
    """
    p_data = np.zeros_like(data, dtype=np.int32)
    count = data.shape[0]
    arr_rowsum = []
    for k in range(1, count // 2 + 1):
        row_sum = 0
        for i in range(k, count - k):
            if data[i] > data[i - k] and data[i] > data[i + k]:
                row_sum -= 1
        arr_rowsum.append(row_sum)
    min_index = np.argmin(arr_rowsum)
    max_window_length = min_index
    for k in range(1, max_window_length + 1):
        for i in range(k, count - k):
            if data[i] > data[i - k] and data[i] > data[i + k]:
                p_data[i] += 1
    return np.where(p_data == max_window_length)[0]


def AMPD_test(path):
    # path = r'D:\my_projects_V1\my_projects\PPG_V1\data\real_data\wu\20221205103726.txt'
    # path = r'D:\my_projects_V1\my_projects\pyqt\learnDemo\data\2022_5_3_16_3.txt'
    ir2, red2 = read_data(path, label="real")
    but_ir2 = bandpass_filter(ir2, start_fs=0.1, end_fs=5)
    peaks = AMPD(but_ir2)
    print(peaks)
    peaks_y = [but_ir2[x] for x in peaks]
    plt.figure(figsize=(10, 8))
    plt.subplot(211)
    plt.plot(ir2)
    plt.title("ir2")
    plt.subplot(212)
    plt.plot(but_ir2)

    plt.scatter(peaks, peaks_y, marker='*', color="red", label="peak")

    # plt.axhline(th1, color='purple')
    # plt.axhline(ir2_mean, color="green")
    plt.legend(loc="best")
    plt.title("PPG")
    plt.show()


def lms(x, d, N=4, mu=0.05):
    L = min(len(x),len(d))
    h = np.zeros(N)
    e = np.zeros(L-N)
    for n in range(L-N):
        x_n = x[n:n+N][::-1]
        d_n = d[n]
        y_n = np.dot(h, x_n.T)
        e_n = d_n - y_n
        h = h + mu * e_n * x_n
        e[n] = e_n
    return e

def lms_test():
    path = r'D:\my_projects_V1\my_projects\PPG_V1\data\real_data\wu\20221205103726.txt'
    ir2, red2 = read_data(path, label="real")
    but_ir2 = bandpass_filter(ir2, start_fs=0.1, end_fs=5)
    e = lms(but_ir2, but_ir2)
    print(len(e))

def detect_peaks(x, mph=None, mpd=1, threshold=0, edge='rising',
                 kpsh=False, valley=False, show=False, ax=None):
    """Detect peaks in data based on their amplitude and other features.
    Parameters
    ----------
    x : 1D array_like
        data.
    mph : {None, number}, optional (default = None)
        detect peaks that are greater than minimum peak height.
    mpd : positive integer, optional (default = 1)
        detect peaks that are at least separated by minimum peak distance (in
        number of data).
    threshold : positive number, optional (default = 0)
        detect peaks (valleys) that are greater (smaller) than `threshold`
        in relation to their immediate neighbors.
    edge : {None, 'rising', 'falling', 'both'}, optional (default = 'rising')
        for a flat peak, keep only the rising edge ('rising'), only the
        falling edge ('falling'), both edges ('both'), or don't detect a
        flat peak (None).
    kpsh : bool, optional (default = False)
        keep peaks with same height even if they are closer than `mpd`.
    valley : bool, optional (default = False)
        if True (1), detect valleys (local minima) instead of peaks.
    show : bool, optional (default = False)
        if True (1), plot data in matplotlib figure.
    ax : a matplotlib.axes.Axes instance, optional (default = None).
    Returns
    -------
    ind : 1D array_like
        indeces of the peaks in `x`.
    Notes
    -----
    The detection of valleys instead of peaks is performed internally by simply
    negating the data: `ind_valleys = detect_peaks(-x)`

    The function can handle NaN's
    See this IPython Notebook [1]_.
    References
    ----------
    "Marcos Duarte, https://github.com/demotu/BMC"
    [1] http://nbviewer.ipython.org/github/demotu/BMC/blob/master/notebooks/DetectPeaks.ipynb
    """

    x = np.atleast_1d(x).astype('float64')
    if x.size < 3:
        return np.array([], dtype=int)
    if valley:
        x = -x
    # find indexes of all peaks
    dx = x[1:] - x[:-1]
    # handle NaN's
    indnan = np.where(np.isnan(x))[0]
    if indnan.size:
        x[indnan] = np.inf
        dx[np.where(np.isnan(dx))[0]] = np.inf
    ine, ire, ife = np.array([[], [], []], dtype=int)
    if not edge:
        ine = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) > 0))[0]
    else:
        if edge.lower() in ['rising', 'both']:
            ire = np.where((np.hstack((dx, 0)) <= 0) & (np.hstack((0, dx)) > 0))[0]
        if edge.lower() in ['falling', 'both']:
            ife = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) >= 0))[0]
    ind = np.unique(np.hstack((ine, ire, ife)))
    # handle NaN's
    if ind.size and indnan.size:
        # NaN's and values close to NaN's cannot be peaks
        ind = ind[np.in1d(ind, np.unique(np.hstack((indnan, indnan - 1, indnan + 1))), invert=True)]
    # first and last values of x cannot be peaks
    if ind.size and ind[0] == 0:
        ind = ind[1:]
    if ind.size and ind[-1] == x.size - 1:
        ind = ind[:-1]
    # remove peaks < minimum peak height
    if ind.size and mph is not None:
        ind = ind[x[ind] >= mph]
    # remove peaks - neighbors < threshold
    if ind.size and threshold > 0:
        dx = np.min(np.vstack([x[ind] - x[ind - 1], x[ind] - x[ind + 1]]), axis=0)
        ind = np.delete(ind, np.where(dx < threshold)[0])
    # detect small peaks closer than minimum peak distance
    if ind.size and mpd > 1:
        ind = ind[np.argsort(x[ind])][::-1]  # sort ind by peak height
        idel = np.zeros(ind.size, dtype=bool)
        for i in range(ind.size):
            if not idel[i]:
                # keep peaks with the same height if kpsh is True
                idel = idel | (ind >= ind[i] - mpd) & (ind <= ind[i] + mpd) \
                       & (x[ind[i]] > x[ind] if kpsh else True)
                idel[i] = 0  # Keep current peak
        # remove the small peaks and sort back the indexes by their occurrence
        ind = np.sort(ind[~idel])

    return ind

def MarcosTest(path):
    # path = r'D:\my_projects_V1\my_projects\PPG_V1\data\real_data\wu\20221205103726.txt'
    # path = r'D:\my_projects_V1\my_projects\pyqt\learnDemo\data\2022_5_3_15_48.txt'
    ir2, red2 = read_data(path, label="real")
    but_ir2 = bandpass_filter(ir2, start_fs=0.1, end_fs=5)

    peaks = detect_peaks(but_ir2, mpd=300)
    plt.plot(but_ir2)
    plt.plot(peaks, but_ir2[peaks], "o")
    plt.title("Marcos Duarte")
    plt.show()

def threshTest(path):
    """阈值法"""

    ir2, red2 = read_data(path, label="real")
    but_ir2 = bandpass_filter(ir2, start_fs=0.1, end_fs=5)
    peaks, _ = scipy.signal.find_peaks(but_ir2, height=(np.max(but_ir2) - np.min(but_ir2)) * 0.1)
    plt.figure(figsize=(10, 8))
    plt.plot(but_ir2, label="PPG")
    plt.plot(peaks, but_ir2[peaks], "o", label="peaks")
    plt.title("幅度阈值法")
    plt.xlabel("采样点")
    plt.ylabel("幅值")
    plt.legend(loc="best")
    plt.savefig(os.path.join(PAPER_FIGURE_PATH, "幅度阈值法"), dpi=200)
    plt.show()


def dwtTest(path):
    ir2, red2 = read_data(path, label="real")
    but_ir2 = bandpass_filter(ir2, start_fs=0.1, end_fs=5)
    peaks = scipy.signal.find_peaks_cwt(but_ir2, np.arange(1, 300))
    plt.plot(but_ir2)
    plt.plot(peaks, but_ir2[peaks], "o")
    plt.title("幅度阈值法")
    plt.xlabel("采样点")
    plt.ylabel("幅值")
    plt.show()

def hrTest(path, frequency=500):
    ir2, red2 = read_data(path, label="real")
    start, end = 0, len(ir2)
    step = 500
    window = 4000
    # window = 2000
    # but_ir2 = bandpass_filter(ir2, start_fs=0.1, end_fs=5)
    while start < end:
        if start+window >= end:
            data = ir2[start: end]
        else:
            data = ir2[start: start+window]
        filtered_data = bandpass_filter(data, start_fs=0.1, end_fs=5)
        # filtered_data = kalman_filter(data)

        peakList = peakTest(filtered_data)
        reversed_data = reverse(filtered_data)
        valleyList = peakTest(reversed_data)
        # print(len(peakList), len(valleyList))
        p = len(peakList)
        hr1 = (p / 8) * 60
        # print("hr1 = ", hr1)
        ppTime = np.diff(peakList)
        intervel_max = max(ppTime)
        intervel_min = min(ppTime)

        # print(ppTime)
        # print(peakList, valleyList)
        hr = ((np.sum(ppTime) - intervel_max - intervel_min) / (len(ppTime)-2)) / frequency * 60
        print("hr =", hr)

        # plt.plot(filtered_data)
        # plt.show()
        if start+window >= end:
            break
        start += step

def hr_plot():
    path = r'D:\my_projects_V1\my_projects\PPG_V1\data\spo2_compare_new\0disturb_3pulse\98\20221130184623_98.txt'  # 模拟
    # path = r'D:\my_projects_V1\my_projects\pyqt\learnDemo\data\2022_5_3_15_48.txt'
    ir2, red2 = read_data(path, label="real")
    data = ir2[1800:5800]
    butter = bandpass_filter(ir2, start_fs=0.1, end_fs=5)
    temp_data = butter[1800:5800]
    peakList = peakTest(temp_data)
    ampList = [temp_data[i] for i in peakList]
    plt.figure(figsize=(10, 8))
    plt.plot(temp_data, label="PPG")
    plt.plot(peakList, ampList, "o", label="peaks", color="purple")
    # plt.title("幅度阈值法")
    plt.xlabel("采样点")
    plt.ylabel("幅值")
    plt.legend(loc="best")
    plt.savefig(os.path.join(PAPER_FIGURE_PATH, "心率计算"), dpi=200)
    plt.show()




if __name__ == '__main__':
    path = r'D:\my_projects_V1\my_projects\pyqt\learnDemo\data\2022_5_3_15_48.txt'   # 真实
    # path = r'D:\my_projects_V1\my_projects\PPG_V1\data\spo2_compare_new\0disturb_3pulse\98\20221130184623_98.txt'   # 模拟
    ir2, red2 = read_data(path, label="real")
    # ir2, red2 = read_data(path, label="simulator")

    peaks(path)
    # AMPD_test(path)
    # MarcosTest(path)
    # dwtTest(path)
    # threshTest(path)
    # hrTest(path)
    # hr_plot()

"""
[  820  1385  1932  2440  2952  3496  4053  4604  5119  5642  6210  6793
  7360  7882  8418  8972  9534 10097 10621 11163 11728 12316 12895 13414
 13954 14521 15100 15648 16192 16777 17339 17886 18413 18951 19506 20037
 20587 21141 21684 22210 22718 23243 23773 24277 24802 25330 25850 26354
 26897 27440 28002 28557 29126 29717]
"""