# -*- coding: utf-8 -*-

"""
@author: Mark Wu
@file: FrequenceAnalysis.py
@time: 2023/3/8 11:44
"""
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score
from sklearn.utils import shuffle

from codes.utils.MyFilters import vmd, bandpass_filter, reverse
from codes.utils.GetFFT import get_freq, my_fft
from codes.PaperTests.BRApneaAnalysis import get_csv_data
from codes.utils.GetFileData import read_from_file
from codes.PaperTests.BRApneaAnalysis import peak_detect, discussion
from codes.PaperTests.Model_Regression import my_random_forest, my_bayes, my_logistic, my_knn, my_svm, my_decision_tree
from codes.PaperTests import PAPER_FIGURE_PATH

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


def get_data(path):
    data = get_csv_data(path)
    ir = data.ir
    resp = data.resp
    pr = data.pr
    spo2 = data.spo2
    return ir, resp, pr, spo2


def get_txt_data():
    path = r"D:\my_projects_V1\my_projects\PPG_V1\data\real_data\wu\apnea\20230223090433.txt"
    data = read_from_file(path)
    ir2 = data.ir2
    resp = data.resp
    plt.figure(figsize=(10, 8))
    plt.subplot(211)
    plt.plot(ir2)
    plt.subplot(212)
    plt.plot(resp)
    plt.show()


def list_split(data, n):
    for i in range(0, len(data), n):
        yield data[i:i + n]


def get_piece_peaks(piece_data, interval=40):
    peakList = []
    max_data = max(piece_data)
    # max_data = np.mean(piece_data)

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
                    # print(peak_index)
                    # if piece_data[peak_index] < 0.25 * max_data or piece_data[peak_index] < 0:
                    if piece_data[peak_index] < 0.25 * max_data:
                        break
                    if len(peakList) == 0:
                        peakList.append(peak_index)
                    else:
                        if peak_index - peakList[-1] < interval:
                            break
                        else:
                            peakList.append(peak_index)

    for i in range(len(piece_data)-6, len(piece_data)):
        # print("j = %s,%s,%s,%s" % (i, piece_data[i-2], piece_data[i-1], piece_data[i]))
        if i+1 < len(piece_data) and piece_data[i] > piece_data[i-1] and piece_data[i] > piece_data[i+1]:
            if piece_data[i] < 0.2 * max_data or i in peakList:
                continue
            else:
                peakList.append(i)

    return peakList


def new_peak_detect1():
    file_path = r"D:\my_projects_V1\my_projects\PPG_V1\data\BR\apnea_csv\wu\20230306110712.csv"
    ir, resp, pr, spo2 = get_data(file_path)
    butter_ir = bandpass_filter(ir, fs=100, start_fs=0.1, end_fs=5)
    data = butter_ir[0:4000]
    init_data = butter_ir[0:500]
    period_diff = []
    period_amp = []
    for i in range(5):
        temp_data = init_data[i*100:(i+1)*100]
        max_diff = max(np.diff(temp_data))
        max_amp = max(temp_data)
        period_diff.append(max_diff)
        period_amp.append(max_amp)
    period_amp.remove(max(period_amp))
    period_diff.remove(max(period_diff))
    c0 = np.mean(period_diff)
    h0 = np.mean(period_amp)
    c_threshold = 0.7 * c0
    h_threshold_low = 0.5 * h0
    h_threshold_high = 1.5 * h0
    peak_list = []
    for i in range(len(data)-2):
        if data[i+1] - data[i] < c_threshold and data[i+2] - data[i+1] < c_threshold:
            index = i+1
            for j in range(4):
                if index+j+3 < len(data) and data[index+j+1] - data[index+j] > 0 and data[index+j+2] - data[index+j+1] > 0 and data[index+j+3] - data[index+j+2] < 0:
                    temp_amp = data[index+j]
                    if h_threshold_low < temp_amp < h_threshold_high:
                        peak_list.append(index+j+2)
    peak_value = [data[x] for x in peak_list]
    plt.plot(data)
    plt.scatter(peak_list, peak_value, color="red")
    plt.show()
    results, u, vmd_resp = vmd(ir)
    buttered_ir = bandpass_filter(ir, fs=100, start_fs=0.1, end_fs=5)


def valley_detect(data, peak_list, interval=40):
    valley_list = []
    data = list(data)
    if peak_list[0] > 0:
        """第一个峰值点不是起始点，那么在这个峰值点前寻找波谷点"""
        temp_valley = data.index(min(data[0:peak_list[0]]))
        if temp_valley != 0:
            flag1 = True
            # for i in range(0, temp_valley):
            #     if data[i] > data[i-1] and data[i] > data[i+1]:
            #         """判断是否存在重搏波"""
            #         flag1 = False
            # if flag1:
            valley_list.append(temp_valley)

    for i in range(len(peak_list)-1):
        valley_list.append(data.index(min(data[peak_list[i]:peak_list[i+1]])))

    if len(data) > peak_list[-1]:
        """在最后一个峰值点至结束，寻找是否存在波谷点"""
        pos_valley = data.index(min(data[peak_list[-1]:]))
        if pos_valley != len(data):
            for j in range(peak_list[-1], len(data)-1):
                if data[j] < data[j-1] and data[j] < data[j+1]:
                    valley_list.append(j)
                    break
    return valley_list


def new_peak_detect(data, freq=100, period_num=3, interval=40):
    # file_path = r"D:\my_projects_V1\my_projects\PPG_V1\data\BR\apnea_csv\wu\20230306110712.csv"
    # ir, resp, pr, spo2 = get_data(file_path)
    # butter_ir = bandpass_filter(ir, fs=100, start_fs=0.1, end_fs=5)
    # data = butter_ir[0:1200]
    # freq = 100
    # num = 3
    period = int(freq * period_num)

    pieces = list_split(data, period)
    peakList = []
    for number, piece in enumerate(pieces):
        piece_peak_list = get_piece_peaks(piece, interval=interval)
        temp_peak_list = [x + number*period for x in piece_peak_list]
        peakList += temp_peak_list
    # peak_value = [data[x] for x in peakList]
    # plt.plot(data)
    # plt.scatter(peakList, peak_value, color="red")
    # plt.show()

    return peakList


def get_features(data):
    butter_start = bandpass_filter(data, fs=100, start_fs=0.5, end_fs=4)
    # start_peak_list = peak_detect(butter_start, 60)
    start_peak_list1 = new_peak_detect(butter_start, period_num=1.5, interval=50)
    start_peak_list2 = new_peak_detect(butter_start, period_num=1.03, interval=50)
    # print(set(start_peak_list1+start_peak_list2))
    start_peak_list = sorted(list(set(start_peak_list1+start_peak_list2)))

    # reverse_start = reverse(butter_start)
    # start_valley_list = peak_detect(reverse_start, 60)
    # start_valley_list = new_peak_detect(reverse_start, period_num=4)
    start_valley_list = valley_detect(butter_start, start_peak_list)
    # print(start_peak_list, len(start_peak_list))
    # print(start_valley_list, len(start_valley_list))

    start_peak_value = [butter_start[x] for x in start_peak_list]
    start_valley_value = [butter_start[y] for y in start_valley_list]

    # plt.figure(figsize=(10, 8))
    # plt.plot(butter_start)
    # plt.scatter(start_peak_list, start_peak_value, color="red")
    # plt.scatter(start_valley_list, start_valley_value, color="red")
    # plt.show()

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
    start_up_time, start_amp_diff = discussion(start_peak_list, start_valley_list, butter_start)
    print("---------------up time------------------")
    print(start_up_time)

    # 上升支时间均值、标准差
    start_up_time_mean = np.mean(start_up_time)
    start_up_time_std = np.std(start_up_time)
    # start_up_time_var = np.var(start_up_time)
    if start_up_time_mean < 0:
        plt.figure(figsize=(10, 8))
        plt.plot(butter_start)
        plt.scatter(start_peak_list, start_peak_value, color="red")
        plt.scatter(start_valley_list, start_valley_value, color="red")
        plt.show()

    # 幅度差均值、标准差
    start_amp_diff_mean = np.mean(start_amp_diff)
    start_amp_diff_std = np.std(start_amp_diff)
    # start_amp_diff_var = np.var(start_amp_diff)

    # 峰峰幅值均值和标准差
    peak_amp = [butter_start[x] for x in start_peak_list]
    amp_mean = np.mean(peak_amp)
    amp_std = np.std(peak_amp)
    # amp_median = np.median(peak_amp)

    # 计算时间
    # ppTime_mean = start_ppTime_mean / 100
    # ppTime_std = start_ppTime_std / 100
    # ppTime_rms = start_ppTime_rms / 100
    # up_time_mean = start_up_time_mean / 100
    # up_time_std = start_up_time_std / 100
    # print("--------------------特征参数----------------------------")
    # print(raw_start_mean, ppTime_mean, ppTime_std, ppTime_rms, start_nn50_num, pnn50, up_time_mean, up_time_std, start_amp_diff_mean, start_amp_diff_std, amp_mean, amp_std)

    # feature_list = (raw_start_mean, start_ppTime_mean, start_ppTime_std, start_ppTime_rms, start_nn50_num, start_up_time_mean, start_up_time_std, start_amp_diff_mean, start_amp_diff_std)
    feature_list = (raw_start_mean, start_ppTime_mean, start_ppTime_std, start_ppTime_rms, start_nn50_num, pnn50, start_up_time_mean, start_up_time_std, start_amp_diff_mean, start_amp_diff_std, amp_mean, amp_std)

    return feature_list


def processs_one(ir):
    ir = np.array(ir)
    results, u, vmd_resp = vmd(ir)
    butter_vmd_resp = bandpass_filter(vmd_resp, start_fs=0.1, end_fs=0.8, fs=100)
    f, absY = my_fft(butter_vmd_resp, freq=100, num=1)
    # print(f)
    freq, max_amp = get_freq(f, absY)
    plt.figure(figsize=(10, 8))
    plt.subplot(211)
    plt.plot(ir)
    plt.subplot(212)
    plt.plot(f, absY)
    plt.scatter(freq, max_amp, color="red")
    plt.show()
    return freq, max_amp


def process_test1():
    file_path = r"D:\my_projects_V1\my_projects\PPG_V1\data\BR\apnea_csv\wu\20230306110712.csv"
    ir, resp, pr, spo2 = get_data(file_path)
    ir = ir[0:4000]
    results, u, vmd_resp = vmd(ir)
    buttered_ir = bandpass_filter(ir, fs=100, start_fs=0.1, end_fs=5)
    f, absY = my_fft(buttered_ir, freq=100, num=2)
    plt.figure(figsize=(10, 8))
    plt.subplot(311)
    plt.plot(ir)
    plt.subplot(312)
    plt.plot(resp)
    plt.subplot(313)
    plt.plot(f, absY)
    plt.show()


def freq_analysis():
    file_path = r"D:\my_projects_V1\my_projects\PPG_V1\data\BR\apnea_csv\wu\20230306110712.csv"
    ir, resp, pr, spo2 = get_data(file_path)
    freq_res = []
    energy_res = []
    start = 0
    window = 800
    step = 100
    end = start + window
    while end < len(ir):
        temp_data = ir[start:end]
        freq, max_amp = processs_one(temp_data)
        freq_res.append(freq)
        energy_res.append(max_amp)
        start += step
        end = start + window
    print(freq_res)
    plt.scatter(freq_res, energy_res)
    plt.show()


def make_lstm_dataset():
    path = r"D:\my_projects_V1\my_projects\PPG_V1\results\lstm_annotation.xlsx"
    dir_path = r"D:\my_projects_V1\my_projects\PPG_V1\data\BR\apnea_csv\all_csv"
    train_x_path = r"D:\my_projects_V1\my_projects\PPG_V1\data\BR\LSTMDataset\train_x"
    train_y_path = r"D:\my_projects_V1\my_projects\PPG_V1\data\BR\LSTMDataset\train_y\train_label.csv"
    test_x_path = r"D:\my_projects_V1\my_projects\PPG_V1\data\BR\LSTMDataset\test_x"
    test_y_path = r"D:\my_projects_V1\my_projects\PPG_V1\data\BR\LSTMDataset\test_y\test_label.csv"
    record = pd.read_excel(path, engine="openpyxl")
    print("len--------", len(record))
    record["file_name"] = record["file_name"].astype(str)
    count = 0
    # rate = 0.3
    # num = int(len(record) * (1 - rate))
    train_y = []
    test_y = []
    for root_path, dir_name, filenames in os.walk(dir_path):
        for filename in filenames:
            file_path = os.path.join(root_path, filename)
            print(file_path)
            data = get_csv_data(file_path)
            f_name = filename.split(".")[0]
            row_detail = record.loc[record.file_name == f_name]
            if len(row_detail) == 0:
                break
            for i in range(len(row_detail)):
                label = row_detail.iloc[i]
                normal_start = label["normal_start"]
                norma_end = label["normal_end"]
                apnea_start = label["apnea_start"]
                apnea_end = label["apnea_end"]
                print(normal_start, norma_end, apnea_start, apnea_end)
                if normal_start == 0 and norma_end == 0:
                    pass
                else:
                    data_normal = data[normal_start:norma_end]
                    save_normal = data_normal[["ir", "pr", "spo2", "rrInterval"]]
                    # print(save_normal)
                    if count <= 40:
                        save_path = os.path.join(train_x_path, str(count)+".csv")
                    else:
                        save_path = os.path.join(test_x_path, str(count)+".csv")
                    print("normal ----- ", save_path)

                    if count <= 40:
                        train_y.append([f_name, 0])
                    else:
                        test_y.append([f_name, 0])
                    save_normal.to_csv(save_path, index=False)
                    count += 1
                if apnea_start == 0 and apnea_end == 0:
                    pass
                else:
                    data_apnea = data[apnea_start:apnea_end]
                    save_apnea = data_apnea[["ir", "pr", "spo2", "rrInterval"]]
                    if count <= 40:
                        save_path = os.path.join(train_x_path, str(count)+".csv")
                    else:
                        save_path = os.path.join(test_x_path, str(count)+".csv")
                    if count <= 40:
                        train_y.append([f_name, 1])
                    else:
                        test_y.append([f_name, 1])
                    save_apnea.to_csv(save_path, index=False)
                    count += 1
                    print("apnea ----- ", save_path)

    df_train_y = pd.DataFrame(train_y, columns=["file_name", "train_label"])
    df_test_y = pd.DataFrame(test_y, columns=["file_name", "test_label"])
    df_train_y.to_csv(train_y_path, index=False)
    df_test_y.to_csv(test_y_path, index=False)


def make_lstm_dataset_all():
    path = r"D:\my_projects_V1\my_projects\PPG_V1\results\lstm_annotation.xlsx"
    dir_path = r"D:\my_projects_V1\my_projects\PPG_V1\data\BR\apnea_csv\all_csv"
    save_dir = r"D:\my_projects_V1\my_projects\PPG_V1\data\BR\LSTMDataset\all_data_modify"
    record = pd.read_excel(path, engine="openpyxl")
    record["file_name"] = record["file_name"].astype(str)
    count = 1
    for root_path, dir_name, filenames in os.walk(dir_path):
        for filename in filenames:
            file_path = os.path.join(root_path, filename)
            print(file_path)
            data = get_csv_data(file_path)
            f_name = filename.split(".")[0]
            row_detail = record.loc[record.file_name == f_name]
            if len(row_detail) == 0:
                break
            for i in range(len(row_detail)):
                label = row_detail.iloc[i]
                normal_start = label["normal_start"]
                norma_end = label["normal_end"]
                apnea_start = label["apnea_start"]
                apnea_end = label["apnea_end"]
                print(normal_start, norma_end, apnea_start, apnea_end)
                if normal_start == 0 and norma_end == 0:
                    pass
                else:
                    data_normal = data[normal_start:norma_end]
                    save_normal = data_normal[["ir", "pr", "spo2", "rrInterval"]]
                    save_path = os.path.join(save_dir, str(count)+"_1.csv")
                    save_normal.to_csv(save_path, index=False)
                    count += 1
                if apnea_start == 0 and apnea_end == 0:
                    pass
                else:
                    data_apnea = data[apnea_start:apnea_end]
                    save_apnea = data_apnea[["ir", "pr", "spo2", "rrInterval"]]
                    save_path = os.path.join(save_dir, str(count)+"_0.csv")
                    save_apnea.to_csv(save_path, index=False)
                    count += 1


def split_data_10():
    dir_path = r"D:\my_projects_V1\my_projects\PPG_V1\data\BR\LSTMDataset\all_data_rever"
    save_dir = r"D:\my_projects_V1\my_projects\PPG_V1\data\BR\LSTMDataset\all_data_10"
    count = 0
    for root_path, dir_name, filenames in os.walk(dir_path):
        for filename in filenames:
            file_path = os.path.join(root_path, filename)
            print(file_path)
            data = get_csv_data(file_path)
            f_name = filename.split(".")[0]
            label = f_name.split("_")[1]

            for i in range(3):
                count += 1
                period_data = data[i*1000:(i+1)*1000].reset_index(drop=True)
                save_path = os.path.join(save_dir, str(count)+"_%s.csv" % label)
                period_data.to_csv(save_path, index=False)



def make_features():
    """制作特征数据集"""
    # dir_path = r"D:\my_projects_V1\my_projects\PPG_V1\data\BR\LSTMDataset\all_data_modify"
    # save_path = r"D:\my_projects_V1\my_projects\PPG_V1\data\BR\LSTMDataset\features_modify.csv"
    dir_path = r"D:\my_projects_V1\my_projects\PPG_V1\data\BR\LSTMDataset\all_data_10"
    save_path = r"D:\my_projects_V1\my_projects\PPG_V1\data\BR\LSTMDataset\features_10_new.csv"

    features = []
    for root_path, dir_name, filenames in os.walk(dir_path):
        for filename in filenames:
            file_path = os.path.join(root_path, filename)
            print(filename)
            label = int(filename.split(".")[0].split("_")[1])
            data = get_csv_data(file_path)
            ir = data.ir
            feature_list = get_features(ir)

            feature = (label,) + feature_list
            features.append(feature)
    column = ["label", "raw_mean", "ppTime_mean", "ppTime_std", "ppTime_rms", "nn50_num", "pnn50", "up_time_mean",
              "up_time_std", "amp_diff_mean", "amp_diff_std", "peak_amp_mean", "peak_amp_std"]
    df_features = pd.DataFrame(features, columns=column)
    # 保存特征
    # df_features.to_csv(save_path, index=False)
    plt.figure(figsize=(10, 8))
    sns.heatmap(df_features.corr(), annot=True)
    plt.xticks(rotation=45)
    # plt.savefig(os.path.join(PAPER_FIGURE_PATH, "相关性new"), dpi=300, bbox_inches="tight")
    plt.show()


def model_test():
    # path = r"D:\my_projects_V1\my_projects\PPG_V1\data\BR\LSTMDataset\features.csv"
    # path = r"D:\my_projects_V1\my_projects\PPG_V1\data\BR\LSTMDataset\features_rever.csv"
    # path = r"D:\my_projects_V1\my_projects\PPG_V1\data\BR\LSTMDataset\features_modify.csv"
    path = r"D:\my_projects_V1\my_projects\PPG_V1\data\BR\LSTMDataset\features_10_new.csv"

    data = pd.read_csv(path, sep=",")
    # model = svm.SVC(kernel="linear", decision_function_shape="ovo")
    # model = RandomForestClassifier()
    # model = KNeighborsClassifier()
    model = LogisticRegression()
    y = data.label
    x = data.iloc[:, 1:]
    x, y = shuffle(x, y)

    train_data, test_data, train_target, test_target = train_test_split(x, y, test_size=0.3, random_state=20, shuffle=False)    # 初始random_state =42
    std = StandardScaler()
    train_data_std = std.fit_transform(train_data)
    test_data_std = std.transform(test_data)

    print(train_data_std.shape, train_target.shape, test_data_std.shape, test_target.shape)
    my_svm(train_data_std, train_target, test_data_std, test_target, pic_save=False, model_save=True, model_save_name="svm12features")
    my_logistic(train_data_std, train_target, test_data_std, test_target, pic_save=False)
    my_random_forest(train_data_std, train_target, test_data_std, test_target, pic_save=False)
    my_knn(train_data_std, train_target, test_data_std, test_target, pic_save=False)
    my_bayes(train_data_std, train_target, test_data_std, test_target, pic_save=False)
    my_decision_tree(train_data_std, train_target, test_data_std, test_target, pic_save=False)


    # model.fit(train_data_std, train_target)
    # # score = model.score(test_data_std, test_target)
    # acu_train = model.score(train_data_std, train_target)
    # acu_test = model.score(test_data_std, test_target)
    # y_pred = model.predict(test_data_std)
    # recall = recall_score(test_target, y_pred, average="macro")
    # print(acu_train, acu_test, recall)


def time_analysis():
    # file_path = r"D:\my_projects_V1\my_projects\PPG_V1\data\BR\apnea_csv\wu\20230306110712.csv"
    # file_path = r"D:\my_projects_V1\my_projects\PPG_V1\data\BR\apnea_csv\good_test1\20230301205746.csv"
    # file_path = r"D:\my_projects_V1\my_projects\PPG_V1\data\BR\short_apnea_csv\wu\30\20230307095931.csv"
    file_path = r"D:\my_projects_V1\my_projects\PPG_V1\data\BR\apnea_csv\all_csv\20230302205202.csv"
    ir, resp, pr, spo2 = get_data(file_path)
    start = 0
    window = 800
    step = 100
    end = start + window
    raw_start_mean = []
    start_ppTime_mean = []
    start_ppTime_std = []
    start_ppTime_rms = []
    start_pnn50_num = []
    start_up_time_mean = []
    start_up_time_std = []
    start_amp_diff_mean = []
    start_amp_diff_std = []
    while end < len(ir):
        print("start = ", start)
        temp_data = ir[start:end]
        features_list = get_features(temp_data)
        raw_start_mean.append(features_list[0])
        start_ppTime_mean.append(features_list[1])
        start_ppTime_std.append(features_list[2])
        start_ppTime_rms.append(features_list[3])
        start_pnn50_num.append(features_list[4])
        start_up_time_mean.append(features_list[5])
        start_up_time_std.append(features_list[6])
        start_amp_diff_mean.append(features_list[7])
        start_amp_diff_std.append(features_list[8])
        start += step
        end = start + window

    plt.figure(figsize=(10, 8))
    plt.subplot(311)
    plt.plot(resp)
    plt.title("呼吸信号")
    plt.subplot(312)
    plt.plot(ir)
    plt.title("PPG信号")
    plt.subplot(313)
    plt.plot(start_amp_diff_mean)
    plt.title("幅度差标准差")
    plt.subplots_adjust(hspace=0.8)
    plt.show()


def model_test1():
    with open("logistic.pickle", "rb") as f:
        clf = pickle.load(f)
    # file_path = r"D:\my_projects_V1\my_projects\PPG_V1\data\BR\apnea_csv\all_csv\20230305102716.csv"
    file_path = r"D:\my_projects_V1\my_projects\PPG_V1\data\BR\apnea_csv\all_csv\20230302205202.csv"
    # file_path = r"D:\my_projects_V1\my_projects\PPG_V1\data\BR\apnea_csv\all_csv\20230301174724.csv"

    ir, resp, pr, spo2 = get_data(file_path)

    start = 0
    window = 1000
    step = 100
    end = start + window
    results = []
    temp = []
    count = 0
    while end < len(ir):
        # print("start = ", start)
        temp_data = ir[start:end]
        features_list = get_features(temp_data)
        temp.append(features_list)
        # arr_feat = np.array(features_list).reshape(1, -1)
        # if count < 5:
        #     temp.append(features_list)
        # else:
        #     df = pd.DataFrame(temp)
        #     res = clf.predict(df.values)
        #     print(res)
        #     count = 0
        #     temp = []
        # print(features_list)
        # res = clf.predict(arr_feat)
        # results.append(res[0])
        start += step
        end = start + window
        # count += 1
    # print(results)
    df = pd.DataFrame(temp)
    test = df.values
    std = StandardScaler()
    test_std = std.fit_transform(test)
    res = clf.predict(test_std)
    print(res, len(res))

    plt.figure(figsize=(10, 8))
    plt.subplot(211)
    plt.plot(ir)
    plt.subplot(212)
    plt.plot(resp)
    plt.show()

if __name__ == '__main__':
    # main()
    # make_lstm_dataset()
    # make_lstm_dataset_all()
    # make_features()
    model_test()
    # model_test1()
    # process_test1()
    # time_analysis()
    # new_peak_detect()

    # 分割成10s的片段
    # split_data_10()
