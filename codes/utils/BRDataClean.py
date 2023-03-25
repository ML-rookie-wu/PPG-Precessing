# -*- coding: utf-8 -*-

"""
@author: Mark Wu
@file: BRDataClean.py
@time: 2023/3/1 20:09
"""
import pandas as pd
import os
import matplotlib.pyplot as plt



def get_raw_data(path, save_dir, start=5000, end=25000):
    raw = pd.read_table(path, sep="\t")
    file_name = os.path.split(path)[1].replace("txt", "csv")
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    save_path = os.path.join(save_dir, file_name)
    if len(raw) < 25000:
        start, end = 0, len(raw)
    data = raw[start:end]
    ir = data[" IIr_Sample"]
    resp = data[" Resp_Sample"]
    pr = data[" PR"]
    spo2 = data[" SPO2"]
    spo2_real = data[" SPO2_Real"]
    rrInterval = data[" rrInterval"]
    df = pd.concat([ir, resp, pr, spo2, spo2_real, rrInterval], axis=1)
    # print(df)
    df.columns = ["ir", "resp", "pr", "spo2", "spo2_real", "rrInterval"]
    df.to_csv(save_path, index=False)
    # os.remove(path)

def modify(csv_path, start=None, end=None, value=None):
    file_name = os.path.split(csv_path)[1].split(".")[0]
    record = pd.read_csv(csv_path, sep=",")
    save_name = file_name + "m"
    save_path = os.path.join(os.path.split(csv_path)[0], save_name)
    print(save_path)
    ir = record.ir
    resp = record.resp
    pr = record.pr
    spo2 = record.spo2
    plt.plot(pr)
    plt.show()

    pr.iloc[2000:3000] = 63

    # plt.figure(figsize=(10, 8))
    plt.plot(pr)
    plt.show()


if __name__ == '__main__':
    # csv_root_path = r"D:\my_projects_V1\my_projects\PPG_V1\data\BR\apnea_csv"

    csv_root_path = r"D:\my_projects_V1\my_projects\PPG_V1\data\BR\short_apnea_csv\wu"

    # csv_dir_name = "wu"
    # csv_dir_name = "tong"
    # csv_dir_name = "luo"
    # csv_dir_name = "lu"
    # csv_dir_name = "hu"
    # csv_dir_name = "xv"

    csv_dir_name = "20"
    # csv_dir_name = "30"
    # csv_dir_name = "40"
    # csv_dir_name = "50"
    # csv_dir_name = "60"

    # path = r"D:\my_projects_V1\my_projects\PPG_V1\data\BR\apnea\wu\20230306112311.txt"
    # path = r"D:\my_projects_V1\my_projects\PPG_V1\data\BR\apnea\tong\20230305104243.txt"
    # path = r"D:\my_projects_V1\my_projects\PPG_V1\data\BR\apnea\luo\20230304103143.txt"
    # path = r"D:\my_projects_V1\my_projects\PPG_V1\data\BR\apnea\hu\20230304215746.txt"
    # path = r"D:\my_projects_V1\my_projects\PPG_V1\data\BR\apnea\lu\20230304103143.txt"
    # path = r"D:\my_projects_V1\my_projects\PPG_V1\data\BR\apnea\xv\20230305110408.txt"

    # 不同时长
    path = r"D:\my_projects_V1\my_projects\PPG_V1\data\BR\short_apnea\wu\20\20230307112534.txt"
    # path = r"D:\my_projects_V1\my_projects\PPG_V1\data\BR\short_apnea\wu\30\20230307095931.txt"
    # path = r"D:\my_projects_V1\my_projects\PPG_V1\data\BR\short_apnea\wu\40\20230307103227.txt"
    # path = r"D:\my_projects_V1\my_projects\PPG_V1\data\BR\short_apnea\wu\50\20230307104909.txt"



    # save_dir = r"D:\my_projects_V1\my_projects\PPG_V1\data\BR\apnea_csv\wu"
    save_dir = os.path.join(csv_root_path, csv_dir_name)
    get_raw_data(path, save_dir)

    # modify(r"D:\my_projects_V1\my_projects\PPG_V1\data\BR\apnea_csv\wu\20230304090439.csv")

