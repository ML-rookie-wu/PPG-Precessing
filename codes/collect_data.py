# -*- coding: utf-86 -*-
# @Time : 2022/10/30 20:54
# @Author : Mark Wu
# @FileName: collect_data.py
# @Software: PyCharm
import glob
import os
import matplotlib.pyplot as plt
from pathlib import Path

import pandas as pd
import serial.tools.list_ports as lp
import serial
import time



def get_port():
    port = list(lp.comports()[0])[0]
    return port

def read_from_port(read_time, dir, spo2_value):
    baud = 921600
    freq = 500
    # spo2 = spo2_value
    port = get_port()
    t = serial.Serial(port, baudrate=baud, timeout=60)
    count = 0
    data_dir = os.path.join(dir, str(spo2_value))
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    path = os.path.join(data_dir, time.strftime("%Y%m%d%H%M%S", time.localtime()) + '_' + str(spo2_value) + '.txt')
    now_time = time.time()
    # print("path------------", path)

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

def collect_data():
    read_time = 10
    # data_dir = os.path.join(Path(__file__).parent.parent, 'data')
    data_dir = r'E:\my_projects\PPG\data\spo2_compare\3disturb_5pulse'
    # print(os.path.splitext(data_dir))
    spo2 = 84
    for i in range(5):
        path = read_from_port(read_time, data_dir, spo2)

def draw(data):
    plt.figure(figsize=(10, 8))
    plt.subplot(211)
    plt.plot(data['ir1'])
    plt.title('ir1')
    plt.subplot(212)
    plt.plot(data['ir2'])
    plt.title('ir2')
    plt.show()

def collect_test():
    baud = 921600
    freq = 500
    # spo2 = spo2_value
    port = get_port()
    print('port-------------', port)
    t = serial.Serial(port, baudrate=baud, timeout=60)
    count = 0
    with open("./read_test_data.txt", "a+") as f:
        while count < 4000:
            try:
                data = t.readline().decode().strip()
                print(data)
                f.write(data + "\n")

            except Exception as e:
                print(e)
                t.readline()
            else:
                count += 1
    data = pd.read_table("./read_test_data.txt", sep=",")
    data.columns = ["ir1", "red1", "ir2", "red2"]
    draw(data)
    os.remove("./read_test_data.txt")

def read_test():
    dir_name = r'E:\my_projects\PPG\data\spo2_compare\2disturb_5pulse\100'
    for path in Path(dir_name).glob("*.txt"):
        print(path)
        data = pd.read_table(path, sep=',')
        data.columns = ["ir1", "red1", "ir2", "red2"]
        draw(data)

def read_from_port_test():
    baud = 921600
    freq = 500
    # spo2 = spo2_value
    port = get_port()
    print('port-------------', port)
    t = serial.Serial(port, baudrate=baud, timeout=60)
    data = t.read(1000)
    print(data)


if __name__ == '__main__':
    # collect_test()
    # collect_data()
    # read_test()

    read_from_port_test()