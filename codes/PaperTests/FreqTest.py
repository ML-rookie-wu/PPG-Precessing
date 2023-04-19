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
import pickle
import pandas as pd
import seaborn as sns
from copy import deepcopy
from scipy import interpolate

from codes.utils.GetFileData import read_from_file
from codes.utils.GetFFT import signal_fft
from codes.utils.MyFilters import bandpass_filter, VMD, vmd
from codes.PaperTests.FrequenceAnalysis import valley_detect
from codes.PaperTests.PeakCompare import new_get_peak
from codes.PaperTests.HampleTest import hampel
from codes.utils.GetFFT import my_fft, get_freq
from codes.PaperTests.Anova import analysis
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler
from codes.PaperTests.Model_Regression import my_random_forest, my_bayes, my_logistic, my_knn, my_svm, my_decision_tree



def interpolation(data, points):
    '''
    三次样条插值实现
    '''
    length = len(data)
    # lenpoints = len(points)

    temp_points = deepcopy(points)
    # temp_points.insert(0, 0)
    # temp_points.append(len(data)-1)

    # temp_data = data[0:points[-1]+1]
    # length = len(temp_data)

    Y = [data[x] for x in temp_points]

    inpolate_points = np.arange(0, length)
    para = interpolate.splrep(temp_points, Y, k=3)
    inpvalue = interpolate.splev(inpolate_points, para)

    return (inpolate_points, inpvalue)

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
                if index + j < len(piece_data) and (piece_data[index + j] - piece_data[index + j - 1]) > 0 and \
                        (piece_data[index + j + 1] - piece_data[index + j]) < 0:
                    peak_index = index + j

                    if piece_data[peak_index] < 0.35 * max_data:
                        # 考虑最大值小于0，piece[peak_index]也小于0
                        if piece_data[peak_index] == max_data:
                            pass
                        elif max_data - piece_data[peak_index] < 2000:
                            pass
                        else:
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


def discussion(peak_list, valley_list, data, peak_valley_interval=50):
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
            if x[1] - x[0] < peak_valley_interval:
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
            if x[1] - x[0] < peak_valley_interval:
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
            if x[1] - x[0] < peak_valley_interval:
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
            if x[1] - x[0] < peak_valley_interval:
                continue
            up_time.append(x[1] - x[0])
            amp_diff.append(data[x[0]] - data[x[1]])
        return up_time, amp_diff
    else:
        raise Exception("the number of peak is not equal with the number of valley")


def discussion1(peak_list, valley_list, peak_valley_interval=100):
    """去除重搏波波峰点和波谷点"""
    if len(valley_list) - len(peak_list) == 1:
        """波谷点比波峰点多一个，valley 开头，valley 结尾"""
        temp_valley_list = copy.deepcopy(valley_list)
        first_valley = temp_valley_list.pop(0)
        combine_list = list(zip(peak_list, temp_valley_list))
        # copy_combine_list = copy.deepcopy(combine_list)
        del_list = []
        for i in range(len(combine_list)):
            if combine_list[i][1] - combine_list[i][0] < peak_valley_interval:
                del_list.append(i)
        if len(del_list) != 0:
            del_list.reverse()
            for index in del_list:
                combine_list.pop(index)
        new_peak_list = [x[0] for x in combine_list]
        new_valley_list = [x[1] for x in combine_list]
        new_valley_list.insert(0, first_valley)
        return new_peak_list, new_valley_list

    elif len(peak_list) - len(valley_list) == 1:
        """波峰点比波谷点多一个，peak 开头，peak 结尾"""
        temp_peak_list = copy.deepcopy(peak_list)
        last_peak = temp_peak_list.pop()
        combine_list = list(zip(peak_list, valley_list))
        # copy_combine_list = copy.deepcopy(combine_list)
        del_list = []
        for i in range(len(combine_list)):
            if combine_list[i][1] - combine_list[i][0] < peak_valley_interval:
                del_list.append(i)

        if len(del_list) != 0:
            del_list.reverse()
            for index in del_list:
                combine_list.pop(index)

        new_peak_list = [x[0] for x in combine_list]
        new_valley_list = [x[1] for x in combine_list]
        new_peak_list.append(last_peak)
        return new_peak_list, new_valley_list

    elif len(peak_list) == len(valley_list) and peak_list[0] < valley_list[0] and peak_list[-1] < valley_list[-1]:
        """波峰点和波谷点一样多，以波峰点开头，波谷点结尾"""
        temp_peak_list = copy.deepcopy(peak_list)
        temp_valley_list = copy.deepcopy(valley_list)
        combine_list = list(zip(temp_peak_list, temp_valley_list))
        del_list = []
        for i in range(len(combine_list)):
            if combine_list[i][1] - combine_list[i][0] < peak_valley_interval:
                del_list.append(i)

        if len(del_list) != 0:
            del_list.reverse()
            for index in del_list:
                combine_list.pop(index)

        new_peak_list = [x[0] for x in combine_list]
        new_valley_list = [x[1] for x in combine_list]
        return new_peak_list, new_valley_list

    elif len(peak_list) == len(valley_list) and peak_list[0] > valley_list[0] and peak_list[-1] > valley_list[-1]:
        """波峰点和波谷点一样多，以波谷点开头，波峰点结尾"""
        temp_peak_list = copy.deepcopy(peak_list)
        temp_valley_list = copy.deepcopy(valley_list)
        last_peak = temp_peak_list.pop()
        first_valley = temp_valley_list.pop(0)
        combine_list = list(zip(temp_peak_list, temp_valley_list))

        del_list = []
        for i in range(len(combine_list)):
            if combine_list[i][1] - combine_list[i][0] < peak_valley_interval:
                del_list.append(i)

        if len(del_list) != 0:
            del_list.reverse()
            for index in del_list:
                combine_list.pop(index)

        new_peak_list = [x[0] for x in combine_list]
        new_valley_list = [x[1] for x in combine_list]
        new_peak_list.append(last_peak)
        new_valley_list.insert(0, first_valley)
        return new_peak_list, new_valley_list
    else:
        raise Exception("the number of peak is not equal with the number of valley")


def new_peak_detect(data, freq=500, period_num=3, interval=350):
    period = int(freq * period_num)
    pieces = list_split(data, period)
    peakList = []
    for number, piece in enumerate(pieces):

        piece_peak_list = get_piece_peaks(piece, interval=interval)
        temp_peak_list = [x + number*period for x in piece_peak_list]
        # print("num = ", number)
        # print("temp_peak_list = ", temp_peak_list)
        peakList += temp_peak_list
    return peakList


def get_features(data, filtered_data):
    start_peak_list1 = new_peak_detect(filtered_data, freq=500, period_num=1.5, interval=400)
    start_peak_list2 = new_peak_detect(filtered_data, freq=500, period_num=0.95, interval=400)
    start_peak_list = sorted(list(set(start_peak_list1+start_peak_list2)))
    start_valley_list = valley_detect(filtered_data, start_peak_list)
    # start_peak_value = [butter_start[x] for x in start_peak_list]
    # start_valley_value = [butter_start[y] for y in start_valley_list]

    peak_list, valley_list = discussion1(start_peak_list, start_valley_list)

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
    start_nn50_num = len([x for x in np.diff(start_ppTime) if x > 5])
    # print("start_pnn50_num =", start_pnn50_num)
    pnn50 = start_nn50_num / len(start_ppTime)

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

    # 峰峰幅值均值和标准差
    peak_amp = [filtered_data[x] for x in start_peak_list]
    amp_mean = np.mean(peak_amp)
    amp_std = np.std(peak_amp)

    # feature_list = (raw_start_mean, start_ppTime_mean, start_ppTime_std, start_ppTime_rms, start_pnn50_num, start_up_time_mean, start_up_time_std, start_amp_diff_mean, start_amp_diff_std)
    feature_list = (raw_start_mean, start_ppTime_mean, start_ppTime_std, start_ppTime_rms, start_nn50_num, pnn50, start_up_time_mean, start_up_time_std, start_amp_diff_mean, start_amp_diff_std, amp_mean, amp_std)

    return feature_list, peak_list, valley_list


def test(dir_path):
    for root_path, dir_name, filenames in os.walk(dir_path):
        for filename in filenames:
            file_path = os.path.join(root_path, filename)
            print(file_path)
            data = read_from_file(file_path)
            ir = data.ir2
            raw_data = ir
            filtered_data = bandpass_filter(ir, fs=500, start_fs=0.1, end_fs=3)[5000:10000]

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
    f9 = []
    f10 = []
    f11 = []
    # with open("svm94.pickle", "rb") as f:
    #     clf = pickle.load(f)
    for root_path, dir_name, filenames in os.walk(dir_path):
        for filename in filenames:
            file_path = os.path.join(root_path, filename)
            if file_path.endswith("csv"):
                continue
            print(file_path)
            data = read_from_file(file_path)
            ir = data.ir2
            raw_data = ir[5000:210000]
            filtered_data = bandpass_filter(ir, fs=500, start_fs=0.1, end_fs=5)[5000:10000]

            # first_diff = np.diff(filtered_data)
            # plt.subplot(211)
            # plt.plot(filtered_data)
            # plt.subplot(212)
            # plt.plot(first_diff)
            # plt.show()
            # filtered_data = bandpass_filter(raw_data, fs=500, start_fs=0.1, end_fs=5)
            feature_list, peak_list, valley_list = get_features(raw_data, filtered_data)

            # df = pd.DataFrame(feature_list)
            # f = df.values
            # std = StandardScaler()
            # f_std = std.fit_transform(f)
            # input = f_std.reshape(1, 9)
            # res = clf.predict(input)
            # print("res = ", res)

            all_feature_list.append(feature_list)
            print(feature_list)
            peak_value = [filtered_data[x] for x in peak_list]
            valley_value = [filtered_data[y] for y in valley_list]

            peak_zip = list(zip(peak_list, peak_value))
            slope_positive = []
            slope_negative = []
            for i in range(len(peak_zip)-1):
                slope = (peak_zip[i+1][1] - peak_zip[i][1]) / (peak_zip[i+1][0]-peak_zip[i][0])
                if slope < 0:
                    slope_negative.append(slope)
                else:
                    slope_positive.append(slope)
            print("slope_positive_mean = ", np.mean(slope_positive))
            print("slope_negative_mean = ", np.mean(slope_negative))

            inpolate_points, inpvalue = interpolation(filtered_data, peak_list)
            butter_inpvalue = bandpass_filter(inpvalue, fs=500, start_fs=0.05, end_fs=0.6)
            f, absY = signal_fft(butter_inpvalue, freq=500)

            plt.subplot(211)
            plt.plot(filtered_data)
            plt.scatter(peak_list, peak_value, color="red")
            plt.scatter(valley_list, valley_value, color="cyan")
            # plt.plot(peak_list, peak_value, color="red")     # 折线图
            plt.plot(inpolate_points, inpvalue, color="red")
            plt.subplot(212)
            plt.plot(f, absY)
            plt.show()

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
        f9.append(all_feature_list[i][9])
        f10.append(all_feature_list[i][10])
        f11.append(all_feature_list[i][11])

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
    plt.subplots_adjust(hspace=0.8)
    plt.show()


def test_make_feature(dir_path):
    features = []
    save_path = r"D:\my_projects_V1\my_projects\PPG_V1\data\respiration2\features1.csv"
    for root_path, dir_name, filenames in os.walk(dir_path):
        for filename in filenames:
            file_path = os.path.join(root_path, filename)
            if file_path.endswith("csv"):
                continue
            print(file_path)
            resp = os.path.split(os.path.dirname(file_path))[1]
            print(resp)
            if resp == "0":
                label = 1
            else:
                label = 0

            data = read_from_file(file_path)
            ir = data.ir2
            butter_ir = bandpass_filter(ir, fs=500, start_fs=0.1, end_fs=3)
            start = 0
            step = 1000
            window = 5000
            end = start + window
            while end < len(ir):
                # print("start = %s, end = %s" % (start, end))
                wind_data = ir[start:end]
                filtered_data = butter_ir[start: end]
                # filtered_data = bandpass_filter(temp_data, fs=500, start_fs=0.1, end_fs=3)
                features_list, peak_list, valley_list = get_features(wind_data, filtered_data)
                # plt.subplot(211)
                # plt.plot(wind_data)
                # plt.subplot(212)
                # plt.plot(filtered_data)
                # plt.show()
                feature_list, peak_list, valley_list = get_features(wind_data, filtered_data)
                feature = (label,) + feature_list
                features.append(feature)

                start += step
                end = start + window
    column = ["label", "raw_mean", "ppTime_mean", "ppTime_std", "ppTime_rms", "nn50_num", "pnn50",
                      "up_time_mean", "up_time_std", "amp_diff_mean", "amp_diff_std", "peak_amp_mean", "peak_amp_std"]
    df_features = pd.DataFrame(features, columns=column)
    # 保存特征
    df_features.to_csv(save_path, index=False)
    plt.figure(figsize=(10, 8))
    sns.heatmap(df_features.corr(), annot=True)
    plt.xticks(rotation=45)
    # plt.savefig(os.path.join(PAPER_FIGURE_PATH, "相关性new"), dpi=300, bbox_inches="tight")
    plt.show()


def data_analysis(csv_path):
    df_0 = pd.read_csv(csv_path, sep=",")
    column_name = df_0.columns
    analysis(df_0, column_name)


def model_test1():
    path = r"D:\my_projects_V1\my_projects\PPG_V1\data\respiration2\features1.csv"

    data = pd.read_csv(path, sep=",")
    # model = svm.SVC(kernel="linear", decision_function_shape="ovo")
    # model = RandomForestClassifier()
    # model = KNeighborsClassifier()
    # model = LogisticRegression()
    y = data.label
    x = data.iloc[:, 1:]
    x, y = shuffle(x, y)

    train_data, test_data, train_target, test_target = train_test_split(x, y, test_size=0.3, random_state=20,
                                                                        shuffle=True)  # 初始random_state =42
    std = StandardScaler()
    train_data_std = std.fit_transform(train_data)
    test_data_std = std.transform(test_data)

    print(train_data_std.shape, train_target.shape, test_data_std.shape, test_target.shape)
    my_svm(train_data_std, train_target, test_data_std, test_target, pic_save=False, model_save=True, model_save_name="simulator_svm")
    my_logistic(train_data_std, train_target, test_data_std, test_target, pic_save=False)
    my_random_forest(train_data_std, train_target, test_data_std, test_target, pic_save=False, model_save=True, model_save_name="simulator_randomForest")
    my_knn(train_data_std, train_target, test_data_std, test_target, pic_save=False, model_save=True, model_save_name="simulator_knn")
    my_bayes(train_data_std, train_target, test_data_std, test_target, pic_save=False)
    my_decision_tree(train_data_std, train_target, test_data_std, test_target, pic_save=False)


def model_test2():
    # model_name = "svm12features.pickle"
    model_name1 = "svm12features.pickle"
    # model_name1 = "simulator_svm.pickle"
    model_name2 = "simulator_randomForest.pickle"
    model_name3 = "simulator_knn.pickle"
    with open(model_name1, "rb") as f:
        clf1 = pickle.load(f)
    with open(model_name2, "rb") as f1:
        clf2 = pickle.load(f1)
    with open(model_name3, "rb") as f2:
        clf3 = pickle.load(f2)
    # file_path = r"D:\my_projects_V1\my_projects\PPG_V1\data\BR\apnea_csv\all_csv\20230305102716.csv"
    # file_path = r"D:\my_projects_V1\my_projects\PPG_V1\data\BR\apnea_csv\all_csv\20230302205202.csv"
    # file_path = r"D:\my_projects_V1\my_projects\PPG_V1\data\BR\apnea_csv\all_csv\20230301174724.csv"
    # file_path = r"D:\my_projects_V1\my_projects\PPG_V1\data\spo2_compare_test\2disturb_3pulse\99\20221128231138_99.txt"
    file_path = r"D:\my_projects_V1\my_projects\PPG_V1\data\respiration2\0\20230320181913.txt"
    # ir, resp, pr, spo2 = get_data(file_path)
    data = read_from_file(file_path)
    ir = data.ir2

    start = 0
    window = 5000
    step = 1000
    end = start + window
    results = []
    temp = []
    count = 0
    butter_ir = bandpass_filter(ir, fs=500, start_fs=0.1, end_fs=3)
    while end < len(ir):
        # print("start = %s, end = %s" % (start, end))
        temp_data = ir[start:end]
        filtered_data = butter_ir[start: end]
        # filtered_data = bandpass_filter(temp_data, fs=500, start_fs=0.1, end_fs=3)
        features_list, peak_list, valley_list = get_features(temp_data, filtered_data)
        temp.append(features_list)

        peak_value = [filtered_data[x] for x in peak_list]
        valley_value = [filtered_data[y] for y in valley_list]

        # plt.plot(filtered_data)
        # plt.scatter(peak_list, peak_value, color="red")
        # plt.scatter(valley_list, valley_value, color="cyan")
        # plt.show()

        start += step
        end = start + window

    # print(results)
    df = pd.DataFrame(temp)
    # print(df)
    test = df.values

    std = StandardScaler()
    test_std = std.fit_transform(test)
    res1 = clf1.predict(test_std)
    res2 = clf2.predict(test_std)
    res3 = clf3.predict(test_std)
    print(res1, len(res1))
    print(res2, len(res2))
    print(res3, len(res3))

    # plt.figure(figsize=(10, 8))
    # plt.subplot(211)
    # plt.plot(ir)
    # plt.subplot(212)
    # plt.plot(resp)
    # plt.show()


if __name__ == '__main__':
    dir_path = r"D:\my_projects_V1\my_projects\PPG_V1\data\respiration2"
    # dir_path = r"D:\my_projects_V1\my_projects\PPG_V1\data\resp_test"
    csv_path = r"D:\my_projects_V1\my_projects\PPG_V1\data\respiration2\features.csv"

    # test(dir_path)
    feature_test(dir_path)
    # test_make_feature(dir_path)
    # data_analysis(csv_path)

    # model_test1()
    # model_test2()

