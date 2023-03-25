# -*- coding: utf-8 -*-

"""
@author: Mark Wu
@file: FreqTest.py
@time: 2023/3/16 11:43
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import copy
from codes.utils.GetFileData import read_from_file
from codes.utils.MyFilters import bandpass_filter, VMD, vmd
from codes.PaperTests.FrequenceAnalysis import valley_detect
from codes.PaperTests.PeakCompare import new_get_peak
from codes.PaperTests.HampleTest import hampel
from codes.utils.GetFFT import my_fft, get_freq



def vmd_test(data, k=9):
    K = k
    alpha = 5000
    tau = 1e-6
    vmd = VMD(K, alpha, tau)
    u, omega_K = vmd(data)
    print(omega_K)
    print(np.mean(omega_K))
    results = data - u[0]
    return results, u, u[0]


def list_split(data, n):
    for i in range(0, len(data), n):
        yield data[i:i + n]


def get_piece_peaks(piece_data, interval=40):
    peakList = []
    max_data = max(piece_data)
    if max_data == piece_data[0] or max_data == piece_data[-1]:
        max_data = np.mean(piece_data)
    diff = np.diff(piece_data)
    for i in range(len(diff) - 2):
        if diff[i] > 0 and diff[i + 1] < 0:
            index = i + 1
            for j in range(0, 4):
                if index + j < len(piece_data) and piece_data[index + j] - piece_data[index + j - 1] > 0 and \
                        piece_data[index + j + 1] - piece_data[index + j] < 0:
                    peak_index = index + j
                    if piece_data[peak_index] < 0.5 * max_data:
                        break
                    if len(peakList) == 0:
                        peakList.append(peak_index)
                    else:
                        if peak_index - peakList[-1] < interval:
                            break
                        else:
                            peakList.append(peak_index)

    # for i in range(len(piece_data)-6, len(piece_data)):
    #     if i+1 < len(piece_data) and piece_data[i] > piece_data[i-1] and piece_data[i] > piece_data[i+1]:
    #         if piece_data[i] < 0.5 * max_data or i in peakList:
    #             continue
    #         else:
    #             peakList.append(i)

    return peakList


def discussion(peak_list, valley_list, data):
    """获取上升支时间、波峰点与波谷点幅度差，根据波峰点和波谷点分四种情况讨论，如果不符合四种情况，说明波峰点或波谷点识别出错"""
    up_time = []
    amp_diff = []
    if len(valley_list) - len(peak_list) == 1:
        """波谷点比波峰点多一个，valley 开头，valley 结尾"""
        temp_valley_list = copy.deepcopy(valley_list)
        temp_valley_list.pop(0)
        combine_list = list(zip(peak_list, temp_valley_list))
        # print("combine1----", combine_list)
        for x in combine_list:
            if x[1] - x[0] < 50:
                continue
            up_time.append(x[1] - x[0])
            amp_diff.append(data[x[0]] - data[x[1]])
        return up_time, amp_diff
    elif len(peak_list) - len(valley_list) == 1:
        """波峰点比波谷点多一个，peak 开头，peak 结尾"""
        temp_peak_list = copy.deepcopy(peak_list)
        temp_peak_list.pop()
        combine_list = list(zip(peak_list, valley_list))
        # print("combine2------", combine_list)
        for x in combine_list:
            if x[1] - x[0] < 50:
                continue
            up_time.append(x[1] - x[0])
            amp_diff.append(data[x[0]] - data[x[1]])
        return up_time, amp_diff
    elif len(peak_list) == len(valley_list) and peak_list[0] < valley_list[0] and peak_list[-1] < valley_list[-1]:
        """波峰点和波谷点一样多，以波峰点开头，波谷点结尾"""
        temp_peak_list = copy.deepcopy(peak_list)
        temp_valley_list = copy.deepcopy(valley_list)
        combine_list = list(zip(temp_peak_list, temp_valley_list))
        # print("combine3-----------", combine_list)
        for x in combine_list:
            if x[1] - x[0] < 50:
                continue
            up_time.append(x[1] - x[0])
            amp_diff.append(data[x[0]] - data[x[1]])
        return up_time, amp_diff
    elif len(peak_list) == len(valley_list) and peak_list[0] > valley_list[0] and peak_list[-1] > valley_list[-1]:
        """波峰点和波谷点一样多，以波谷点开头，波峰点结尾"""
        temp_peak_list = copy.deepcopy(peak_list)
        temp_valley_list = copy.deepcopy(valley_list)
        temp_peak_list.pop()
        temp_valley_list.pop(0)
        combine_list = list(zip(temp_peak_list, temp_valley_list))
        # print("combine4------", combine_list)
        for x in combine_list:
            if x[1] - x[0] < 50:
                continue
            up_time.append(x[1] - x[0])
            amp_diff.append(data[x[0]] - data[x[1]])
        return up_time, amp_diff
    else:
        raise Exception("the number of peak is not equal with the number of valley")


def new_peak_detect(data, freq=500, period_num=3, interval=350):
    period = int(freq * period_num)
    pieces = list_split(data, period)
    peakList = []
    for number, piece in enumerate(pieces):
        piece_peak_list = get_piece_peaks(piece, interval=interval)
        temp_peak_list = [x + number*period for x in piece_peak_list]
        peakList += temp_peak_list
    return peakList


def get_features(data, filtered_data):
    start_peak_list1 = new_peak_detect(filtered_data, freq=500, period_num=1.5, interval=350)
    start_peak_list2 = new_peak_detect(filtered_data, freq=500, period_num=1.03, interval=350)
    start_peak_list = sorted(list(set(start_peak_list1+start_peak_list2)))
    start_valley_list = valley_detect(filtered_data, start_peak_list)
    # start_peak_value = [butter_start[x] for x in start_peak_list]
    # start_valley_value = [butter_start[y] for y in start_valley_list]

    # 原始数据均值
    raw_start_mean = np.mean(data)

    # 峰峰间隔
    start_ppTime = np.diff(start_peak_list)  # pp间期
    start_ppTime_mean = np.mean(start_ppTime)  # pp间期均值
    # print("mean = ", start_ppTime_mean)
    start_ppTime_std = np.std(start_ppTime)  # pp间期标准差
    # start_ppTime_var = np.var(start_ppTime)
    start_ppTime_rms = np.sqrt(sum([x ** 2 for x in start_ppTime]) / len(start_ppTime))  # pp间期均方根
    # print("rms = ", start_ppTime_rms)
    start_pnn50_num = len([x for x in np.diff(start_ppTime) if x > 5])
    # print("start_pnn50_num =", start_pnn50_num)

    # 上升支时间、波峰点和波谷点幅度差
    start_up_time, start_amp_diff = discussion(start_peak_list, start_valley_list, filtered_data)

    # 上升支时间均值、标准差
    start_up_time_mean = np.mean(start_up_time)
    start_up_time_std = np.std(start_up_time)
    # start_up_time_var = np.var(start_up_time)

    # 幅度差均值、标准差
    start_amp_diff_mean = np.mean(start_amp_diff)
    start_amp_diff_std = np.std(start_amp_diff)
    # start_amp_diff_var = np.var(start_amp_diff)
    feature_list = (raw_start_mean, start_ppTime_mean, start_ppTime_std, start_ppTime_rms, start_pnn50_num, start_up_time_mean, start_up_time_std, start_amp_diff_mean, start_amp_diff_std)
    return feature_list, start_peak_list, start_valley_list

def test(dir_path):
    for root_path, dir_name, filenames in os.walk(dir_path):
        for filename in filenames:
            file_path = os.path.join(root_path, filename)
            print(file_path)
            data = read_from_file(file_path)
            ir = data.ir2
            raw_data = ir
            filtered_data = bandpass_filter(ir, fs=500, start_fs=0.1, end_fs=5)[5000:10000]

            # for i in range(3, 15):
            # results, u, vmd_resp = vmd_test(raw_data)
            # restruct = np.sum(u, 0)

            # restruct = np.zeros([len(u[0])])
            # u_list = u.tolist()
            # for x in u_list:
            #     restruct += x
            # restruct = np.sum(u)
            # print(restruct)

            # plt.subplot(211)
            # plt.plot(raw_data)
            # plt.subplot(212)
            # plt.plot(restruct)
            # plt.show()

            butter = bandpass_filter(ir, fs=500, start_fs=0.1, end_fs=0.8)
            f, absY = my_fft(butter, freq=500)
            freq, max_amp = get_freq(f, absY)
            print("freq = %s , amp = %s" % (freq, max_amp))
            features_list, peak_list, valley_list = get_features(raw_data, filtered_data)
            peak_values = [filtered_data[x] for x in peak_list]
            # print(features_list)

            # plt.figure(figsize=(10, 8))
            # plt.subplot(311)
            # plt.plot(ir)
            # plt.subplot(312)
            # plt.plot(filtered_data)
            # plt.scatter(peak_list, peak_values)
            #
            # # plt.subplot(313)
            # # plt.plot(vmd_resp)
            # plt.show()


def feature_test(dir_path):
    all_feature_list = []
    f0 = []
    f1 = []
    f2 = []
    f3 = []
    f4 = []
    f5 = []
    f6 = []
    f7 = []
    f8 = []
    for root_path, dir_name, filenames in os.walk(dir_path):
        for filename in filenames:
            file_path = os.path.join(root_path, filename)
            print(file_path)
            data = read_from_file(file_path)
            ir = data.ir2
            raw_data = ir[5000:9000]
            filtered_data = bandpass_filter(ir, fs=500, start_fs=0.6, end_fs=3)[5000:9000]
            # filtered_data = bandpass_filter(raw_data, fs=500, start_fs=0.1, end_fs=5)
            feature_list, start_peak_list, start_valley_list = get_features(raw_data, filtered_data)
            all_feature_list.append(feature_list)
            print(feature_list)
            peak_value = [filtered_data[x] for x in start_peak_list]
            valley_value = [filtered_data[y] for y in start_valley_list]

            # plt.plot(filtered_data)
            # plt.scatter(start_peak_list, peak_value, color="red")
            # plt.scatter(start_valley_list, valley_value, color="cyan")
            # plt.show()

    for i in range(len(all_feature_list)):
        f0.append(all_feature_list[i][0])
        f1.append(all_feature_list[i][1])
        f2.append(all_feature_list[i][2])
        f3.append(all_feature_list[i][3])
        f4.append(all_feature_list[i][4])
        f5.append(all_feature_list[i][5])
        f6.append(all_feature_list[i][6])
        f7.append(all_feature_list[i][7])
        f8.append(all_feature_list[i][8])

    plt.subplot(911)
    plt.plot(f0)
    plt.subplot(912)
    plt.plot(f1)
    plt.subplot(913)
    plt.plot(f2)
    plt.subplot(914)
    plt.plot(f3)
    plt.subplot(915)
    plt.plot(f4)
    plt.subplot(916)
    plt.plot(f5)
    plt.subplot(917)
    plt.plot(f6)
    plt.subplot(918)
    plt.plot(f7)
    plt.subplot(919)
    plt.plot(f8)
    plt.show()

if __name__ == '__main__':
    dir_path = r"D:\my_projects_V1\my_projects\PPG_V1\data\respiration1"
    # dir_path = r"D:\my_projects_V1\my_projects\PPG_V1\data\resp_test"
    test(dir_path)
    # feature_test(dir_path)



