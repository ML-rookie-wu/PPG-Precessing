# -*- coding: utf-8 -*-

"""
@author: Mark Wu
@file: BRApneaAnalysis.py
@time: 2023/3/3 13:47
"""

import pandas as pd
import os
import copy
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


from codes.utils.MyFilters import bandpass_filter, vmd, reverse
from codes.utils.GetFFT import get_freq, signal_fft
from codes.PaperTests import PAPER_FIGURE_PATH
from codes.PaperTests.Model import svr, mlpregressor, random_forest, adboost, bagging, decision_tree
from codes.PaperTests.HampleTest import hampel

# plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['font.sans-serif'] = ['SongNTR']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


def get_csv_data(path):
    data = pd.read_csv(path, sep=",")
    return data


def my_fft(data, freq=500, num=1):
    N = len(data)
    fs = freq
    df = fs / (N*num - 1)  # df分辨率
    # 构建频率数组
    f = [df * n for n in range(0, N)]
    Y = np.fft.fft(data) * 2 / N
    absY = [np.abs(x) for x in Y]

    return f, absY


def peak_detect(filtered_data, interval):
    peakList = []
    # ir2_max = np.max(filtered_data)
    # ir2_min = np.min(filtered_data)
    # ir2_mean = (sum(filtered_data) - ir2_max - ir2_min) / (len(filtered_data) - 2)

    max_data = abs(max(filtered_data))

    diff = np.diff(filtered_data)
    # diff_max = np.max(diff)
    # diff_min = np.min(diff)
    # diff_mean = (sum(diff) - diff_max - diff_min) / (len(diff) - 2)
    # th1 = 1.2 * diff_mean
    # normal_ir2 = normalization(filtered_data)
    # normal_diff = normalization(diff)
    for i in range(len(diff) - 2):
        if diff[i] > 0 and diff[i + 1] < 0:
            index = i + 1
            for j in range(0, 4):
                if index + j < len(filtered_data) and filtered_data[index + j] - filtered_data[index + j - 1] > 0 and \
                        filtered_data[index + j + 1] - filtered_data[index + j] < 0:
                    peak_index = index + j
                    # print(peak_index)
                    if abs(filtered_data[peak_index]) < 0.23 * max_data or filtered_data[peak_index] < 0:
                        break
                    if len(peakList) == 0:
                        peakList.append(peak_index)
                    else:
                        if peak_index - peakList[-1] < interval:
                            break
                        else:
                            peakList.append(peak_index)
    return peakList


def get_resp_points(resp_data):
    start_point = None
    end_point = None

    return start_point, end_point


def get_pr_points(pr_data):
    pass


def get_window_data(file_name, ir_data, window=800):
    """
    获取心率开始增加前8s窗口和开始下降前8s窗口的PPG数据，以及根据呼吸波记录的呼吸暂停时长
    """
    path = r"D:\my_projects_V1\my_projects\PPG_V1\results\apnea_annotation.xlsx"
    f_name = file_name.split(".")[0]
    record = pd.read_excel(path, engine="openpyxl")
    record["file_name"] = record["file_name"].astype(str)
    row_detail = record.loc[record.file_name == f_name]
    pr_start = row_detail["pr_start"].values[0]
    pr_end = row_detail["pr_end"].values[0]
    spo2_start_down = row_detail["spo2_start_down"].values[0]
    spo2_start_up = row_detail["spo2_start_up"].values[0]
    apnea_start = row_detail["apnea_start"].values[0]
    apnea_end = row_detail["apnea_end"].values[0]
    apnea_time = apnea_end - apnea_start

    start_window = ir_data[pr_start - window: pr_start]
    end_window = ir_data[pr_end - window: pr_end]

    pr_up_delay = pr_start - apnea_start
    pr_down_delay = pr_end - apnea_end
    spo2_down_delay = spo2_start_down - apnea_start
    spo2_up_delay = spo2_start_up - apnea_end
    delay_time = [pr_up_delay, pr_down_delay, spo2_down_delay, spo2_up_delay]
    return start_window, end_window, apnea_time, delay_time


def freq_test(dir_path):
    path = r"D:\my_projects_V1\my_projects\PPG_V1\results\apnea_annotation.xlsx"
    record = pd.read_excel(path, engine="openpyxl")
    record["file_name"] = record["file_name"].astype(str)
    window = 1200
    for root_path, dir_name, filenames in os.walk(dir_path):
        for filename in filenames:
            file_path = os.path.join(root_path, filename)
            print(file_path)
            data = get_csv_data(file_path)
            ir = data.ir
            resp = data.resp
            pr = data.pr
            spo2 = data.spo2
            ir_data = ir.tolist()
            f_name = filename.split(".")[0]
            row_detail = record.loc[record.file_name == f_name]
            pr_start = row_detail["pr_start"].values[0]
            pr_end = row_detail["pr_end"].values[0]
            apnea_start = row_detail["apnea_start"].values[0]
            apnea_end = row_detail["apnea_end"].values[0]
            spo2_start_down = row_detail["spo2_start_down"].values[0]
            spo2_start_up = row_detail["spo2_start_up"].values[0]
            spo2_end_up = row_detail["spo2_end_up"].values[0]

            start_wind = ir_data[apnea_start - window:apnea_start]
            end_wind = ir_data[apnea_end - window:apnea_end]

            start_low_butter = bandpass_filter(start_wind, fs=100, start_fs=0.04, end_fs=0.15)
            start_high_butter = bandpass_filter(end_wind, fs=100, start_fs=0.15, end_fs=0.4)
            start_low_f, start_low_absY = my_fft(start_low_butter, freq=100, num=5)
            start_low_freq, start_low_max_amp = get_freq(start_low_f, start_low_absY)
            start_high_f, start_high_absY = my_fft(start_high_butter, freq=100, num=5)
            start_high_freq, start_high_max_amp = get_freq(start_high_f, start_high_absY)
            print("----------------start----------------")
            print("low_freq=%s, low_max_amp=%s, high_freq=%s, high_max_amp=%s" % (
            start_low_freq, start_low_max_amp, start_high_freq, start_high_max_amp))

            end_low_butter = bandpass_filter(end_wind, fs=100, start_fs=0.04, end_fs=0.15)
            end_high_butter = bandpass_filter(end_wind, fs=100, start_fs=0.15, end_fs=0.4)
            end_low_f, end_low_absY = my_fft(end_low_butter, freq=100, num=5)
            end_low_freq, end_low_max_amp = get_freq(end_low_f, end_low_absY)
            end_high_f, end_high_absY = my_fft(end_high_butter, freq=100, num=5)
            end_high_freq, end_high_max_amp = get_freq(end_high_f, end_high_absY)
            print("------------------end----------------")
            print("low_freq=%s, low_max_amp=%s, high_freq=%s, high_max_amp=%s" % (
            end_low_freq, end_low_max_amp, end_high_freq, end_high_max_amp))



def get_features_points(dir_path):
    path = r"D:\my_projects_V1\my_projects\PPG_V1\results\apnea_annotation.xlsx"
    record = pd.read_excel(path, engine="openpyxl")
    record["file_name"] = record["file_name"].astype(str)
    for root_path, dir_name, filenames in os.walk(dir_path):
        for filename in filenames:
            file_path = os.path.join(root_path, filename)
            print(file_path)
            data = get_csv_data(file_path)
            ir = data.ir
            resp = data.resp
            pr = data.pr
            spo2 = data.spo2
            ir_data = ir.tolist()
            f_name = filename.split(".")[0]
            row_detail = record.loc[record.file_name == f_name]
            pr_start = row_detail["pr_start"].values[0]
            pr_end = row_detail["pr_end"].values[0]
            apnea_start = row_detail["apnea_start"].values[0]
            apnea_end = row_detail["apnea_end"].values[0]
            spo2_start_down = row_detail["spo2_start_down"].values[0]
            spo2_start_up = row_detail["spo2_start_up"].values[0]
            spo2_end_up = row_detail["spo2_end_up"].values[0]

            plt.figure(figsize=(10, 8))
            plt.subplot(411)
            plt.plot(spo2)
            plt.scatter(spo2_start_down, spo2[spo2_start_down], color="red")
            plt.scatter(spo2_start_up, spo2[spo2_start_up], color="red")
            plt.scatter(spo2_end_up, spo2[spo2_end_up], color="red")
            plt.title("血氧饱和度")
            plt.subplots_adjust(hspace=0.8)
            plt.subplot(412)
            plt.plot(resp)
            plt.scatter(apnea_start, resp[apnea_start], color="red")
            plt.scatter(apnea_end, resp[apnea_end], color="red")
            plt.title("呼吸波信号")
            plt.subplots_adjust(hspace=0.8)
            plt.subplot(413)
            plt.plot(pr)
            plt.scatter(pr_start, pr[pr_start], color="crimson")
            plt.scatter(pr_end, pr[pr_end], color="crimson")
            plt.title("脉率")
            plt.subplots_adjust(hspace=0.8)
            plt.subplot(414)
            plt.plot(ir_data)
            plt.title("PPG信号")
            # plt.xlabel("采样点")
            # plt.ylabel("幅值")
            plt.subplots_adjust(hspace=0.8)
            if f_name == "20230301174724":
                plt.savefig(os.path.join(PAPER_FIGURE_PATH, "标注1"), dpi=300, bbox_inches="tight")
            plt.show()


def plot_test(dir_path):
    """查看每条数据波形，人工标注特征点"""
    files = []
    for root_path, dir_name, filenames in os.walk(dir_path):
        for filename in filenames:
            file_path = os.path.join(root_path, filename)
            print(file_path)
            data = get_csv_data(file_path)
            ir = data.ir
            resp = data.resp
            pr = data.pr
            spo2 = data.spo2

            plt.subplot(211)
            plt.plot(ir)
            plt.subplot(212)
            plt.plot(resp)
            plt.show()

            plt.subplot(411)
            plt.plot(pr)
            plt.subplot(412)
            plt.plot(spo2)
            plt.subplot(413)
            plt.plot(resp)
            plt.subplot(414)
            plt.plot(ir)
            plt.show()


def plot_one():
    # file_path = r"D:\my_projects_V1\my_projects\PPG_V1\data\BR\apnea_csv\wu\20230305102716.csv"
    file_path = r"D:\my_projects_V1\my_projects\PPG_V1\data\BR\apnea_csv\wu\20230301174724.csv"
    # file_path = r"D:\my_projects_V1\my_projects\PPG_V1\data\BR\apnea_csv\wu\20230301205746.csv"

    data = get_csv_data(file_path)
    ir = data.ir
    resp = data.resp
    pr = data.pr
    spo2 = data.spo2

    # plt.subplot(211)
    # plt.plot(ir)
    # plt.subplot(212)
    # plt.plot(resp)
    # plt.show()

    x_ticks = np.arange(0, len(pr) + 1000, 2500)
    label = ["0", "12500", "25000", "37500", "50000", "62500", "75000", "87500", "100000"]
    plt.subplot(411)
    plt.plot(pr)
    plt.xticks(x_ticks, label)
    plt.subplot(412)
    plt.plot(spo2)
    plt.xticks(x_ticks, label)
    plt.subplot(413)
    plt.plot(resp)
    plt.xticks(x_ticks, label)
    plt.subplot(414)
    plt.plot(ir)
    plt.xticks(x_ticks, label)
    plt.subplots_adjust(hspace=0.8)
    plt.savefig(os.path.join(PAPER_FIGURE_PATH, "标注modify"), dpi=300, bbox_inches="tight")
    plt.show()


def get_points(data):
    pr = data.pr
    resp = data.resp
    ir = data.ir
    spo2 = data.spo2
    butter_resp = bandpass_filter(resp, fs=100, start_fs=0.1, end_fs=2)


def discussion(peak_list, valley_list, data):
    """获取上升支时间、波峰点与波谷点幅度差，根据波峰点和波谷点分四种情况讨论，如果不符合四种情况，说明波峰点或波谷点识别出错"""
    up_time = []
    amp_diff = []
    if len(valley_list) - len(peak_list) == 1:
        """波谷点比波峰点多一个，valley 开头，valley 结尾"""
        temp_valley_list = copy.deepcopy(valley_list)
        temp_valley_list.pop(0)
        combine_list = list(zip(peak_list, temp_valley_list))
        for x in combine_list:
            up_time.append(x[1] - x[0])
            amp_diff.append(data[x[0]] - data[x[1]])
        return up_time, amp_diff
    elif len(peak_list) - len(valley_list) == 1:
        """波峰点比波谷点多一个，peak 开头，peak 结尾"""
        temp_peak_list = copy.deepcopy(peak_list)
        temp_peak_list.pop()
        combine_list = list(zip(peak_list, valley_list))
        for x in combine_list:
            up_time.append(x[1] - x[0])
            amp_diff.append(data[x[0]] - data[x[1]])
        return up_time, amp_diff
    # elif len(peak_list) == len(valley_list) and data[peak_list[0] - 1] < data[peak_list[0]] and data[peak_list[0] + 1] < \
    #         data[peak_list[0]]:
    elif len(peak_list) == len(valley_list) and peak_list[0] < valley_list[0] and peak_list[-1] < valley_list[-1]:
        """波峰点和波谷点一样多，以波峰点开头，波谷点结尾"""
        temp_peak_list = copy.deepcopy(peak_list)
        temp_valley_list = copy.deepcopy(valley_list)
        combine_list = list(zip(temp_peak_list, temp_valley_list))
        for x in combine_list:
            up_time.append(x[1] - x[0])
            amp_diff.append(data[x[0]] - data[x[1]])
        return up_time, amp_diff
    # elif len(peak_list) == len(valley_list) and data[valley_list[0] - 1] > data[valley_list[0]] and data[
    #     valley_list[0] + 1] > data[valley_list[0]]:
    elif len(peak_list) == len(valley_list) and peak_list[0] > valley_list[0] and peak_list[-1] > valley_list[-1]:
        """波峰点和波谷点一样多，以波谷点开头，波峰点结尾"""
        temp_peak_list = copy.deepcopy(peak_list)
        temp_valley_list = copy.deepcopy(valley_list)
        temp_peak_list.pop()
        temp_valley_list.pop(0)
        combine_list = list(zip(temp_peak_list, temp_valley_list))
        for x in combine_list:
            up_time.append(x[1] - x[0])
            amp_diff.append(data[x[0]] - data[x[1]])
        return up_time, amp_diff
    else:
        raise Exception("the number of peak is not equal with the number of valley")


def get_delay_time():
    delay_list = []
    path = r"D:\my_projects_V1\my_projects\PPG_V1\results\apnea_annotation.xlsx"
    record = pd.read_excel(path, engine="openpyxl")
    pr_start = record.pr_start.tolist()
    pr_end = record.pr_end.tolist()
    spo2_start_down = record.spo2_start_down.tolist()
    apnea_start = record.apnea_start.tolist()
    for i in range(len(record)):
        delay_list.append([(pr_start[i]-apnea_start[i])/100, (spo2_start_down[i]-apnea_start[i])/100])
    df = pd.DataFrame(delay_list)
    df.columns = ["pr_start_delay", "spo2_start_delay"]
    delay_save_path = r"D:\my_projects_V1\my_projects\PPG_V1\results\first_delay_time.csv"
    df.to_csv(delay_save_path, index=False)


def process(dir_path):
    """标注特征点，并获取特征"""
    features = []
    delay_list = []
    for root_path, dir_name, filenames in os.walk(dir_path):
        for filename in filenames:
            file_path = os.path.join(root_path, filename)
            print(file_path)
            data = get_csv_data(file_path)
            ir = data.ir
            resp = data.resp
            pr = data.pr
            spo2 = data.spo2
            ir_data = ir.tolist()
            start_wind, end_wind, apnea_time, delay_time = get_window_data(filename, ir_data, window=1200)
            delay_list.append(delay_time)
            butter_end = bandpass_filter(end_wind, fs=100, start_fs=0.1, end_fs=5)
            butter_start = bandpass_filter(start_wind, fs=100, start_fs=0.1, end_fs=5)
            # results, imfs, vmd_resp = vmd(np.array(end_wind))

            # start_low_butter = bandpass_filter(start_wind, fs=100, start_fs=0.04, end_fs=0.15)
            # start_high_butter = bandpass_filter(end_wind, fs=100, start_fs=0.15, end_fs=0.4)
            # start_low_f, start_low_absY = signal_fft(start_low_butter, freq=100)
            # start_low_freq, start_low_max_amp = get_freq(start_low_f, start_low_absY)
            # start_high_f, start_high_absY = signal_fft(start_high_butter, freq=100)
            # start_high_freq, start_high_max_amp = get_freq(start_high_f, start_high_absY)
            # print("----------------start----------------")
            # print("low_freq=%s, low_max_amp=%s, high_freq=%s, high_max_amp=%s" % (start_low_freq, start_low_max_amp, start_high_freq, start_high_max_amp))
            #
            # end_low_butter = bandpass_filter(end_wind, fs=100, start_fs=0.04, end_fs=0.15)
            # end_high_butter = bandpass_filter(end_wind, fs=100, start_fs=0.15, end_fs=0.4)
            # end_low_f, end_low_absY = signal_fft(end_low_butter, freq=100)
            # end_low_freq, end_low_max_amp = get_freq(end_low_f, end_low_absY)
            # end_high_f, end_high_absY = signal_fft(end_high_butter, freq=100)
            # end_high_freq, end_high_max_amp = get_freq(end_high_f, end_high_absY)
            # print("------------------end----------------")
            # print("low_freq=%s, low_max_amp=%s, high_freq=%s, high_max_amp=%s" % (end_low_freq, end_low_max_amp, end_high_freq, end_high_max_amp))

            start_peak_list = peak_detect(butter_start, 60)
            reverse_start = reverse(butter_start)
            start_valley_list = peak_detect(reverse_start, 60)
            start_peak_value = [butter_start[x] for x in start_peak_list]
            start_valley_value = [butter_start[y] for y in start_valley_list]

            end_peak_list = peak_detect(butter_end, 35)
            reverse_end = reverse(butter_end)
            end_valley_list = peak_detect(reverse_end, 35)
            end_peak_value = [butter_end[i] for i in end_peak_list]
            end_valley_value = [butter_end[j] for j in end_valley_list]

            plt.figure(figsize=(10, 8))
            plt.subplot(211)
            plt.plot(butter_start)
            plt.scatter(start_peak_list, start_peak_value, color="red")
            plt.scatter(start_valley_list, start_valley_value, color="red")
            plt.subplot(212)
            plt.plot(butter_end)
            plt.scatter(end_peak_list, end_peak_value, color="red")
            plt.scatter(end_valley_list, end_valley_value, color="red")
            plt.show()

            # 原始数据均值
            raw_start_mean = np.mean(start_wind)
            raw_end_mean = np.mean(end_wind)

            # 原始数据均值的比值
            raw_mean_ratio = raw_start_mean / raw_end_mean

            # 峰峰间隔
            start_ppTime = np.diff(start_peak_list)     # pp间期
            start_ppTime_mean = np.mean(start_ppTime)   # pp间期均值
            start_ppTime_std = np.std(start_ppTime)     # pp间期标准差
            # start_ppTime_var = np.var(start_ppTime)
            start_ppTime_rms = np.sqrt(sum([x**2 for x in start_ppTime]) / len(start_ppTime))    # pp间期均方根
            start_pnn50_num = len([x for x in np.diff(start_ppTime) if x > 5])
            print("start_pnn50_num =", start_pnn50_num)

            end_ppTime = np.diff(end_peak_list)
            end_ppTime_mean = np.mean(end_ppTime)
            end_ppTime_std = np.std(end_ppTime)
            # end_ppTime_var = np.var(end_ppTime)
            end_ppTime_rms = np.sqrt(sum([x**2 for x in end_ppTime]) / len(end_ppTime))
            end_pnn50_num = len([x for x in np.diff(end_ppTime) if x > 5])
            print("end_pnn50_num =", end_pnn50_num)
            pnn50_diff = end_pnn50_num - start_pnn50_num

            # 峰峰间隔的比值
            ppTime_mean_ratio = start_ppTime_mean / end_ppTime_mean
            ppTime_std_ratio = start_ppTime_std / end_ppTime_std
            ppTime_rms_ratio = start_ppTime_rms / end_ppTime_rms

            # 上升支时间、波峰点和波谷点幅度差
            start_up_time, start_amp_diff = discussion(start_peak_list, start_valley_list, butter_start)
            end_up_time, end_amp_diff = discussion(end_peak_list, end_valley_list, butter_end)

            # 上升支时间均值、标准差
            start_up_time_mean = np.mean(start_up_time)
            end_up_time_mean = np.mean(end_up_time)
            start_up_time_std = np.std(start_up_time)
            end_up_time_std = np.std(end_up_time)
            # start_up_time_var = np.var(start_up_time)
            # end_up_time_var = np.var(end_up_time)

            # 上升支比值
            up_time_mean_ratio = start_up_time_mean / end_up_time_mean
            up_time_std_ratio = start_up_time_std / end_up_time_std

            # 幅度差均值、标准差
            start_amp_diff_mean = np.mean(start_amp_diff)
            end_amp_diff_mean = np.mean(end_amp_diff)
            start_amp_diff_std = np.std(start_amp_diff)
            end_amp_diff_std = np.std(end_amp_diff)
            # start_amp_diff_var = np.var(start_amp_diff)
            # end_amp_diff_var = np.var(end_amp_diff)

            # 幅度差比值
            end_amp_diff_mean_ratio = start_amp_diff_mean / end_amp_diff_mean
            end_amp_diff_std_ratio = start_amp_diff_std / end_amp_diff_std

            print(
                "raw_start_mean=%s\nraw_end_mean=%s\nraw_mean_ratio=%s\nstart_ppTime_mean=%s\nend_ppTime_mean=%s\nppTime_mean_ratio=%s\nstart_ppTime_std=%s\n"
                "end_ppTime_std=%s\nppTime_std_ratio=%s\nstart_ppTime_rms=%s\nend_ppTime_rms=%s\nppTime_rms_ratio=%s\nstart_pnn50_num=%s\nend_pnn50_num=%s\n"
                "start_up_time_mean=%s\nend_up_time_mean=%s\nup_time_mean_ratio=%s\nstart_up_time_std=%s\nend_up_time_std=%s\nup_time_std_ratio=%s\n"
                "start_amp_diff_mean=%s\nend_amp_diff_mean=%s\napm_diff_mean_ratio=%s\nstart_amp_diff_std=%s\nend_amp_diff_std=%s\namp_diff_std_ratio=%s" %
                (raw_start_mean, raw_end_mean, raw_mean_ratio, start_ppTime_mean, end_ppTime_mean, ppTime_mean_ratio, start_ppTime_std, end_ppTime_std, ppTime_std_ratio,
                 start_ppTime_rms, end_ppTime_rms, ppTime_rms_ratio, start_pnn50_num, end_pnn50_num, start_up_time_mean, end_up_time_mean, up_time_mean_ratio, start_up_time_std, end_up_time_std, up_time_std_ratio, start_amp_diff_mean,
                 end_amp_diff_mean, end_amp_diff_mean_ratio, start_amp_diff_std, end_amp_diff_std, end_amp_diff_std_ratio))

            plt.subplot(411)
            plt.plot(start_wind)
            plt.subplot(412)
            plt.plot(butter_start)
            plt.scatter(start_peak_list, start_peak_value, color="red")
            plt.scatter(start_valley_list, start_valley_value, marker="*", color="cyan")
            plt.subplot(413)
            plt.plot(end_wind)
            plt.subplot(414)
            plt.plot(butter_end)
            plt.scatter(end_peak_list, end_peak_value, color="red")
            plt.scatter(end_valley_list, end_valley_value, marker="*", color="cyan")

            plt.show()

            features.append(
                [apnea_time, raw_start_mean, raw_end_mean, raw_mean_ratio, start_ppTime_mean, end_ppTime_mean, ppTime_mean_ratio, start_ppTime_std,
                 end_ppTime_std, ppTime_std_ratio, start_ppTime_rms, end_ppTime_rms, ppTime_rms_ratio, pnn50_diff, start_up_time_mean, end_up_time_mean, up_time_mean_ratio, start_up_time_std, end_up_time_std, up_time_std_ratio,
                 start_amp_diff_mean, end_amp_diff_mean, end_amp_diff_mean_ratio, start_amp_diff_std, end_amp_diff_std, end_amp_diff_std_ratio])
    df_delay = pd.DataFrame(delay_list)
    df_delay.columns = ["pr_up_delay", "pr_down_delay", "spo2_down_delay", "spo2_up_delay"]
    print(df_delay)

    df_features = pd.DataFrame(features)
    df_features.columns = ["f%s" % x for x in range(len(features[0]))]
    # feature_save_path = r"D:\my_projects_V1\my_projects\PPG_V1\results\features_all.csv"
    feature_save_path = r"D:\my_projects_V1\my_projects\PPG_V1\results\features.csv"
    # df_features.to_csv(feature_save_path, index=False)

    plt.figure(figsize=(10, 8))
    sns.heatmap(df_features.corr(), annot=True)
    # plt.savefig(os.path.join(PAPER_FIGURE_PATH, "相关性25"), dpi=300, bbox_inches="tight")
    plt.show()


def identify_features(dir_path):
    for root_path, dir_name, filenames in os.walk(dir_path):
        for filename in filenames:
            file_path = os.path.join(root_path, filename)
            print(file_path)
            data = get_csv_data(file_path)
            ir = data.ir
            resp = data.resp
            pr = data.pr
            spo2 = data.spo2
            ir_data = ir.tolist()
            start = 0
            window = 2000
            step = 500
            end = start + window
            while end < len(ir):
                temp_data = ir[start:end]
                butter_start = bandpass_filter(temp_data, fs=100, start_fs=0.5, end_fs=3)
                start_peak_list = peak_detect(butter_start, 60)
                reverse_start = reverse(butter_start)
                start_valley_list = peak_detect(reverse_start, 60)
                start_peak_value = [butter_start[x] for x in start_peak_list]
                start_valley_value = [butter_start[y] for y in start_valley_list]

                plt.figure(figsize=(10, 8))
                plt.plot(butter_start)
                plt.scatter(start_peak_list, start_peak_value, color="red")
                plt.scatter(start_valley_list, start_valley_value, color="red")
                plt.show()
                start += step
                end = start + window


def model_test():
    path = r"D:\my_projects_V1\my_projects\PPG_V1\results\features_all.csv"
    data = pd.read_csv(path, sep=",")

    y = data.f0
    x = data.iloc[:, 1:]
    # print(y)
    svr(x, y)
    decision_tree(x, y)
    random_forest(x, y)
    adboost(x, y)
    bagging(x, y)
    mlpregressor(x, y)


if __name__ == '__main__':
    # dir_path = r"D:\my_projects_V1\my_projects\PPG_V1\data\BR\apnea_csv\wu"
    # dir_path = r""
    # dir_path = r"D:\my_projects_V1\my_projects\PPG_V1\data\BR\apnea_csv\corr_good"
    # dir_path = r"D:\my_projects_V1\my_projects\PPG_V1\data\BR\apnea_csv\good_test1"
    dir_path = r"D:\my_projects_V1\my_projects\PPG_V1\data\BR\apnea_csv\all_csv"

    # plot_one()
    # plot_test(dir_path)
    # identify_features(dir_path)
    # get_features_points(dir_path)     #
    # freq_test(dir_path)
    process(dir_path)
    # model_test()
    # get_delay_time()
