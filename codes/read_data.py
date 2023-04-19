# -*- coding: utf-86 -*-
# @Time : 2022/9/12 19:00
# @Author : Mark Wu
# @FileName: read_data.py
# @Software: PyCharm
import datetime
import glob
import json
import time
import matplotlib.pyplot as plt
import serial
import serial.tools.list_ports as lp
import os
from pathlib import Path
import pandas as pd
# from codes.process import VMD, bandpass_filter
import numpy as np
import math
from codes.utils.MyFilters import vmd, get_dwt_res, bandpass_filter


def get_port():
    port = list(lp.comports()[0])[0]
    return port

class SerialPort:
    def __init__(self, baud):
        self.baud = baud
        self.t = serial.Serial(get_port(), baudrate=self.baud, timeout=60)
        self.data_dir = os.path.join(Path(__file__).parent.parent, 'data')

    def write_to_txt(self, q):
        save_path = os.path.join(self.data_dir, time.strftime("%Y%m%d%H%M%S", time.localtime()) + '.txt')
        with open(save_path, 'a+') as f:
            while True:
                pass


def read_from_port(read_time, dir, spo2_value):
    baud = 921600
    freq = 500
    # spo2 = spo2_value
    port = get_port()
    print('port-------------', port)
    t = serial.Serial(port, baudrate=baud, timeout=60)
    count = 0
    data_dir = os.path.join(dir, str(spo2_value))
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    path = os.path.join(data_dir, time.strftime("%Y%m%d%H%M%S", time.localtime()) + '_' + str(spo2_value) + '.txt')
    now_time = time.time()
    print(now_time)
    with open(path, 'a+') as f:
        # while time.time()-now_time < read_time:
        while count < 30000:
            try:
                data = t.readline().decode().strip()
                print(data)

                f.write(data + '\n')

            except Exception as e:
                print(e)
                t.readline()
            else:
                count += 1
            #     print(count)
    return path

def read_test():
    baud = 921600
    freq = 500
    # spo2 = spo2_value
    port = get_port()
    print('port-------------', port)
    t = serial.Serial(port, baudrate=baud, timeout=60)
    count = 0
    while count < 30000:
        try:
            data = t.readline().decode().strip()
            print(data)

        except Exception as e:
            print(e)
            t.readline()
        else:
            count += 1


def read_from_file(path):
    data = pd.read_table(path, sep=',')
    # print(type(data))
    data.columns = ['ir1', 'red1', 'ir2', 'red2']

    return data

def collect_data():
    read_time = 10
    data_dir = os.path.join(Path(__file__).parent.parent, 'data')
    print(os.path.splitext(data_dir))
    spo2 = 86
    for i in range(10):
        path = read_from_port(read_time, data_dir, spo2)

def draw(data):
    plt.figure(figsize=(10, 8))
    plt.subplot(211)
    plt.plot(data['ir2'])
    plt.title('ir2')
    plt.subplot(212)
    plt.plot(data['red2'])
    plt.title('red2')
    plt.show()

def normalization(data):
    x_mean = np.mean(data)
    std = np.std(data)
    normaled = [(x - x_mean) / std for x in data]
    return normaled

# def vmd(data):
#     K = 86
#     alpha = 5000
#     tau = 1e-6
#     vmd = VMD(K, alpha, tau)
#     u, omega_K = vmd(data)
#     results = data - u[0]
#     return results, u[0]

def CalSQI():
    data_dir = Path(os.path.join(Path(__file__).parent.parent, 'data'))
    for path in data_dir.glob("*.txt"):
        print(path)
        data = read_from_file(path)
        ir2 = data['ir2'][4000:8000]
        filtered_vmd = vmd(ir2)
        filtered_band = bandpass_filter(ir2)
        y_mean = np.mean(filtered_vmd)
        y_std = np.std(filtered_vmd)
        ir2 = normalization(ir2)
        x_mean = np.mean(ir2)
        x_std = np.std(ir2)
        N = len(ir2)

        z_mean = np.mean(filtered_band)
        z_std = np.std(filtered_band)

        #
        S_x = (1 / N) * sum([((x - x_mean) / x_std) ** 3 for x in ir2])
        S_y = (1 / N) * sum([((x - y_mean) / y_std) ** 3 for x in filtered_vmd])
        S_z = (1 / N) * sum([((x - z_mean) / z_std) ** 3 for x in filtered_band])

        # S_x = (1 / N) * sum([(x - x_mean / x_std) ** 3 for x in ir2])
        # S_y = (1 / N) * sum([(x - y_mean / y_std) ** 3 for x in filtered_vmd])
        # S_z = (1 / N) * sum([(x - z_mean / z_std) ** 3 for x in filtered_band])

        print("S_x = ", S_x)
        print("S_y = ", S_y)
        print("S_z = ", S_z)

def spo2_validate(value):
    value = round(value)
    if value > 100:
        value = 100
    return value

def cal_spo2(R):
    # 三元回归
    # spo2_third = -115.46 + 735.1 * R - 809.24 * (R ** 2) + 281.71 * (R ** 3)

    # spo2_third = 101.26 + 29.5 * R - 100.56 * (R ** 2) + 36.59 * (R ** 3)      # 3disturb_5pulse_test

    spo2_third = 93.29 + 58.64 * R - 118.46 * (R ** 2) + 47.62 * (R ** 3)    # all_results
    spo2_third = spo2_validate(spo2_third)

    # 二元回归
    # spo2_second = 99.39 + fast.13 * R - 26.91 * (R ** 2)

    # spo2_second = 111.2 - 17.54 * R - 13.99 * (R ** 2)         # 3disturb_5pulse_test

    spo2_second = 110.59 - 16.63 * R - 13.07 * (R ** 2)         # all_results
    spo2_second = spo2_validate(spo2_second)

    # 一元回归
    # spo2 = 122.69 - 30.45 * R

    # spo2 = 117.15 - 36.08 * R         # 3disturb_5pulse_test

    spo2 = 117.25 - 35.74 * R         # all_results
    spo2 = spo2_validate(spo2)

    return spo2_third, spo2_second, spo2


class Process:
    def __init__(self, data_dir, save_path=None):
        # self.data_dir = os.path.join(Path(__file__).parent.parent, 'data')
        self.data_dir = data_dir
        self.path_name = self.get_all_file()
        # self.results_path = os.path.join(Path(__file__).parent.parent, 'results')
        self.save_path = save_path

    def get_all_file(self):
        files = []
        # for path_tuple in os.walk(self.data_dir):
        #     if not path_tuple[1]:
        #         for file in path_tuple[2]:
        #             files.append(os.path.join(path_tuple[0], file))

        for filepath, dirpath, filenames in os.walk(self.data_dir):
            for filename in filenames:
                # if filename.find("_copy") >= 0:
                #     files.append(os.path.join(filepath, filename))
                files.append(os.path.join(filepath, filename))

        return files

    @staticmethod
    def read_from_file(path):
        right_data = []
        with open(path, "r") as f:
            raw_data = f.readlines()
            for one in raw_data:
                temp_data = one.strip().replace(" ", "")
                if len(temp_data) == 29:
                    right_data.append([int(x) for x in temp_data.split(",")])
                elif len(temp_data) > 29:
                    right_data.append([int(x) for x in temp_data[:29].split(",")])
                else:
                    pass
        data = pd.DataFrame(right_data)
        data.columns = ["ir1", "red1", "ir2", "red2"]

        # data = pd.read_table(path, sep=",")
        # if data.shape[1] == 4:
        #     data.columns = ["ir1", "red1", "ir2", "red2"]
        # elif data.shape[1] == 5:
        #     data.columns = ["ir1", "red1", "ir2", "red2", "resp"]
        return data["ir2"], data["red2"]

    @staticmethod
    def preprocess(data):
        # K = 86
        # alpha = 5000
        # tau = 1e-6
        # vmd = VMD(K, alpha, tau)
        # u, omega_K = vmd(data)
        # results = data - u[0]

        # results, u, u[0] = vmd(data)
        buttered = bandpass_filter(data, start_fs=0.1, end_fs=5)

        ac, dc = get_dwt_res(buttered)
        resp_disturb = ac[-2][0:len(data)]
        removed = [buttered[i] - resp_disturb[i] for i in range(len(data))]

        # return results, u[0]
        return removed

    @staticmethod
    def get_R(ir, red, ir_baseline, red_baseline):
        winsize = len(ir)
        # mean_ir = np.mean(ir)
        # mean_red = np.mean(red)
        # # 去除直流，直流定义为红光获红外平均值
        # windata_ir_ac = [_ - mean_ir for _ in ir]
        # windata_red_ac = [_ - mean_red for _ in red]
        # # 求红光和红外的平方和，根据平方和求交流，直流交流处理方式
        # ir_ac_pow = [math.pow(_, 2) for _ in windata_ir_ac]
        # sum_ir_ac = np.sum(ir_ac_pow)
        # ir_ac = math.sqrt(sum_ir_ac / winsize)
        #
        # red_ac_pow = [math.pow(_, 2) for _ in windata_red_ac]
        # sum_red_ac = np.sum(red_ac_pow)
        # red_ac = math.sqrt(sum_red_ac / winsize)

        # 直流
        ir_dc = np.mean(ir_baseline)
        red_dc = np.mean(red_baseline)

        # 交流
        ir_ac = math.sqrt(sum(x ** 2 for x in ir)) / winsize
        red_ac = math.sqrt(sum(y ** 2 for y in red)) / winsize


        # 求R值
        # R = (red_ac * mean_ir) / (ir_ac * mean_red)

        R = (red_ac * ir_dc) / (ir_ac * red_dc)

        return R

    def save_to_excel(self):
        # file_name = "R.xlsx"
        # path = os.path.join(self.results_path, file_name)
        if self.save_path is not None:
            writer = pd.ExcelWriter(self.save_path, engine="openpyxl")
            results = self.deal()
            df = pd.DataFrame(results)
            df.columns = ['R', 'spo2_value']
            df.to_excel(writer, index=False)
            writer.save()
        else:
            raise Exception("Need a save path,but there is not")

    def deal(self):
        res = []
        for path in self.path_name:
            print(os.path.split(path)[1])
            spo2_value = int(os.path.split(path)[1].split(".")[0].split("_")[1])
            ir2, red2 = self.read_from_file(path)
            start = 1000
            step = 500
            window = 4000
            end = start + step
            while end <= len(ir2):
                # vmd
                # vmded_ir2, ir2_disturb = self.preprocess(ir2[start: end])
                # vmded_red2, red2_disturb = self.preprocess(red2[start: end])
                # R = self.get_R(vmded_ir2, vmded_red2, ir2_disturb, red2_disturb)

                # 小波
                dwted_ir2 = self.preprocess(ir2[start: end])
                dwted_red2 = self.preprocess(red2[start: end])
                R = self.get_R(dwted_ir2, dwted_red2, ir2, red2)

                print("R = ", R)

                res.append([R, spo2_value])
                start += step
                end = start + window
        return res

    def run(self):
        # file_name = "R1.xlsx"
        self.save_to_excel()

def test(data_dir=None, save_path=None):
    p = Process(data_dir, save_path)
    res = {}
    if data_dir is None:
        data_dir = p.data_dir
    all_dir = list(os.walk(data_dir))
    files = all_dir[0][2]
    cur_dir = all_dir[0][0]

    for file_name in files:
        if file_name.find("txt") < 0:
            continue
        print(file_name)
        real_spo2 = int(file_name.split('_')[-1].split(".")[0])
        if real_spo2 <= 30:
            real_spo2 = 97

        res[real_spo2] = []
        regr = []
        bivariate_regr = []
        ternary_regr = []
        file_path = os.path.join(cur_dir, file_name)
        ir2, red2 = p.read_from_file(file_path)

        start = 5000
        step = 4000
        end = start + step
        while end <= len(ir2):
            vmded_ir2, ir2_baseline = p.preprocess(ir2[start: end])
            vmded_red2, red2_baseline = p.preprocess(red2[start: end])
            R = p.get_R(vmded_ir2, vmded_red2, ir2_baseline, red2_baseline)
            spo2_third, spo2_second, spo2 = cal_spo2(R)
            regr.append(spo2)
            bivariate_regr.append(spo2_second)
            ternary_regr.append(spo2_third)
            start += 1000
            end = start + step
        res[real_spo2].append(regr)
        res[real_spo2].append(bivariate_regr)
        res[real_spo2].append(ternary_regr)
    print(res)
    if save_path is not None:
        with open(save_path, 'w') as f:
            json.dump(res, f)

def SQI_test():
    p = Process()
    res = {}

    all_dir = list(os.walk(p.data_dir))
    files = all_dir[0][2]
    cur_dir = all_dir[0][0]

    for file_name in files:
        print(file_name)
        file_path = os.path.join(cur_dir, file_name)
        ir2, red2 = p.read_from_file(file_path)
        # start = 5000
        # step = 4000
        # end = start + step
        # while end <= len(ir2):
        #     raw_data = ir2[start: end]
        #     vmded_ir2 = p.preprocess(ir2[start: end])
        #     # vmded_red2 = p.preprocess(red2[start: end])
        #     y_max = max(vmded_ir2)
        #     y_min = min(vmded_ir2)
        #     x_mean = np.mean(raw_data)
        #     P_SQI = (y_max - y_min) / x_mean
        #     print("SQI = ", P_SQI)
        #     start += 1000
        #     end = start + step
        raw_data = ir2
        vmded_ir2 = p.preprocess(ir2)
        # vmded_red2 = p.preprocess(red2[start: end])
        y_max = max(vmded_ir2)
        y_min = min(vmded_ir2)
        x_mean = np.mean(raw_data)
        P_SQI = (y_max - y_min) / x_mean
        print("SQI = ", P_SQI)

def test_diff_spo2(data_dir=None, save_path=None):
    p = Process()
    res = {}
    regr = []
    bivariate_regr = []
    ternary_regr = []
    if data_dir is None:
        data_dir = p.data_dir
    all_dir = list(os.walk(data_dir))
    files = all_dir[0][2]
    cur_dir = all_dir[0][0]

    for file_name in files:
        if file_name.find("txt") < 0:
            continue
        print(file_name)
        file_path = os.path.join(cur_dir, file_name)
        ir2, red2 = p.read_from_file(file_path)
        start = 4000
        end = 6000
        vmded_ir2, ir2_baseline = p.preprocess(ir2[start: end])
        vmded_red2, red2_baseline = p.preprocess(red2[start: end])
        R = p.get_R(vmded_ir2, vmded_red2, ir2_baseline, red2_baseline)
        spo2_third, spo2_second, spo2 = cal_spo2(R)
        regr.append(spo2)
        bivariate_regr.append(spo2_second)
        ternary_regr.append(spo2_third)
    res["first"] = regr
    res["second"] = bivariate_regr
    res["third"] = ternary_regr
    print(res)
    if save_path is not None:
        with open(save_path, 'w') as f:
            json.dump(res, f)

def cal_R():
    data_dir = r'E:\my_projects\PPG\data\spo2_compare\3disturb_5pulse'
    excel_name = os.path.split(data_dir)[1] + ".xlsx"
    excel_path = os.path.join(r'E:\my_projects\PPG\results\R_value_new', excel_name)

    process = Process(data_dir, save_path=excel_path)
    process.run()

def read_file_test():
    data_dir = r'E:\my_projects\PPG\data\spo2_compare\0disturb_4pulse'
    files = []
    for path_tuple in os.walk(data_dir):
        print(path_tuple)
        if not path_tuple[1]:
            for file in path_tuple[2]:
                files.append(os.path.join(path_tuple[0], file))
    for path in files:
        # print(path)
        try:
            data = pd.read_table(path, sep=",")
            data.columns = ["ir1", "red1", "ir2", "red2"]
            # data_list = data.values.tolist()
            # for one in data_list:
            #     temp = "".join(list(map(str, one)))
            #     if len(temp) != 26:
            #         print(one)

        except Exception as e:
            print(e)


def read_file_test_copy():
    data_dir = r'E:\my_projects\PPG\data\spo2_compare\1disturb_5pulse'
    files = []
    for path_tuple in os.walk(data_dir):
        if not path_tuple[1]:
            for file in path_tuple[2]:
                files.append(os.path.join(path_tuple[0], file))

    for path in files:

        print(path)
        try:
            right_data = []
            with open(path, "r") as f:
                data = f.readlines()
                for one in data:
                    temp_data = one.strip().replace(" ", "")
                    if len(temp_data) == 29:
                        right_data.append(temp_data + "\n")
                        # right_data.append(x.strip() for x in temp_data.split(","))
                    elif len(temp_data) > 29:
                        # right_data.append([int(x.strip()) for x in temp_data[:29].split(",")])
                        print("--------------------", temp_data[:29])
                        break
                    else:
                        pass
            os.remove(path)
            with open(path, "a+") as f_write:
                for one in right_data:
                    f_write.write(one)

        except Exception as e:
            print(e)

        # df = pd.DataFrame(right_data)
        # df.columns = ["ir1", "red1", "ir2", "red2"]
        # draw(df)

def modify_error_file(dir_path):
    for filepath, dirnames, filenames in os.walk(dir_path):
        if len(filenames) >= 1 and filenames[0].endswith("txt"):
            for filename in filenames:
                file = os.path.join(filepath, filename)
                filename_split = filename.split('.')
                new_file_name = filename_split[0] + "_copy." + filename_split[1]
                new_file_path = os.path.join(filepath, new_file_name)
                if os.path.exists(new_file_path):
                    os.remove(new_file_path)
                print(new_file_path)
                right_data = []
                with open(file, 'r') as f:
                    data = f.readlines()
                    for one in data:
                        temp_data = one.strip().replace(" ", "")
                        if len(temp_data) == 29:
                            right_data.append(temp_data + "\n")
                            # right_data.append(x.strip() for x in temp_data.split(","))
                        elif len(temp_data) > 29:
                            # right_data.append([int(x.strip()) for x in temp_data[:29].split(",")])
                            # print("--------------------", temp_data[:29])
                            right_data.append(temp_data[:29] + '\n')
                        else:
                            pass

                with open(new_file_path, "a+") as f_write:
                    for one in right_data:
                        f_write.write(one)

def read_all(dir_path):
    for filepath, dirnames, filenames in os.walk(dir_path):
        if len(filenames) >= 1 and filenames[0].endswith("txt"):
            for filename in filenames:
                file = os.path.join(filepath, filename)

                # print(os.path.basename(file))
                if file.find("copy") >= 0:
                    print(file)
                    data = read_from_file(file)
                    plt.figure(figsize=(10, 8))
                    plt.subplot(211)
                    plt.plot(data.ir2)
                    plt.subplot(212)
                    plt.plot(data.red2)
                    plt.show()

def remove_file(dir_path):
    for filepath, dirnames, filenames in os.walk(dir_path):
        if len(filenames) >= 1 and filenames[0].endswith("txt"):
            for filename in filenames:
                file = os.path.join(filepath, filename)
                if file.find("_copy") >= 0:
                    os.remove(file)

if __name__ == '__main__':
    # read_file_test()     # 先进行检查
    cal_R()

    # a = os.path.join(Path(__file__).parent.parent, 'data')
    # print(list(os.walk(a)))

    # CalSQI()

    # test(data_dir=r"E:\my_projects\PPG\data\test_data\3disturb_3pulse", save_path=r"E:\my_projects\PPG\results\cal_json\3disturb_3pulse.json")

    # read_test()

    # SQI_test()

    # dir_path = r'E:\my_projects\PPG\data\spo2_compare\1disturb_5pulse'
    # modify_error_file(dir_path)
    # read_all(dir_path)
    # remove_file(dir_path)