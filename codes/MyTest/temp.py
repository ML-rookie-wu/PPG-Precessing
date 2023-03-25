# -*- coding: utf-8 -*-

"""
@author: Mark Wu
@file: temp.py
@time: 2023/3/8 15:43
"""

import serial.tools.list_ports as lp
import os
import time
import serial
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd


def get_port():
    # print(list(lp.comports()[0]))
    port = list(lp.comports()[0])[0]
    # port = "COM8"

    return port

def read_from_port(dir, spo2_value):
    baud = 115200
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

    return path

def Byte_to_int(data_byte):
    ir1 = int.from_bytes(data_byte[0:3], byteorder='big')
    red1 = int.from_bytes(data_byte[3:6], byteorder='big')
    ir2 = int.from_bytes(data_byte[6:9], byteorder='big')
    red2 = int.from_bytes(data_byte[9:12], byteorder='big')
    resp = int.from_bytes(data_byte[12:14], byteorder='big')

    return f'{ir1},{red1},{ir2},{red2},{resp}\n'

def crc_check_next(crc, crc_next):
    # crc = binascii.b2a_hex(crc).decode('utf-8')
    # test_crc = int(crc, 16)  # 将str类型转化成16进制
    test_crc = crc
    test_next = int(crc_next, 16)
    test_crc = test_crc ^ test_next
    ploy = '0x07'
    poly_crc8 = int(ploy, 16)  # 将str类型转化成16进制
    for bit in range(8):
        if (test_crc & 0x80) != 0:  # 最高位是否为1
            test_crc <<= 1
            test_crc &= 0xff
            test_crc ^= poly_crc8
        else:
            test_crc <<= 1
    return hex(test_crc)

def crc_check(crc):
    # crc = binascii.b2a_hex(crc)
    # test_crc = int(crc, 16)  # 将str类型转化成16进制
    test_crc = crc
    ploy = '0x07'
    poly_crc8 = int(ploy, 16)  # 将str类型转化成16进制
    # poly_crc8 = 0x07
    for bit in range(8):
        if (test_crc & 0x80) != 0:  # 最高位是否为1, 0x80:128
            # print(test_crc & 0x80)
            test_crc <<= 1
            test_crc &= 0xff    # 0xff:255
            test_crc ^= poly_crc8
        else:
            test_crc <<= 1
    return hex(test_crc)

def judge_crc(data_byte):
    crc_next0 = crc_check(data_byte[0])
    crc_next1 = crc_check_next(data_byte[1], crc_next0)
    crc_next2 = crc_check_next(data_byte[2], crc_next1)
    crc_next3 = crc_check_next(data_byte[3], crc_next2)
    crc_next4 = crc_check_next(data_byte[4], crc_next3)
    crc_next5 = crc_check_next(data_byte[5], crc_next4)
    crc_next6 = crc_check_next(data_byte[6], crc_next5)
    crc_next7 = crc_check_next(data_byte[7], crc_next6)
    crc_next8 = crc_check_next(data_byte[8], crc_next7)
    crc_next9 = crc_check_next(data_byte[9], crc_next8)
    crc_next10 = crc_check_next(data_byte[10], crc_next9)
    crc_next11 = crc_check_next(data_byte[11], crc_next10)
    crc_next12 = crc_check_next(data_byte[12], crc_next11)
    crc_next13 = crc_check_next(data_byte[13], crc_next12)

    int_crc_next11 = int(crc_next13, 16)

    return int_crc_next11


def read_package(path, num=30000):
    port = get_port()
    baud = 115200
    s = serial.Serial(port, baudrate=baud, timeout=60)
    print("port---------------", port)
    with open(path, 'w') as file:
        total = 0
        while total < num:
            # i += 1
            first_byte = s.read()
            if first_byte == b'\xaa':
                second_byte = s.read()
                if second_byte == b'\xaa':
                    data_byte = s.read(14)   # b'\x1f\xff\x19\x1f\xff\x19\x1f\xff\x19\x1f\xff\x19\xc9\xfa'
                    crc_final = judge_crc(data_byte)  # 计算crc
                    crc_data = s.read().hex()  # 十六进制字符串，真实的crc
                    last_byte = s.read(2)
                    if last_byte == b'\x04\n':
                        content = Byte_to_int(data_byte)
                        # content = str(time.time()) + "," + content
                        print(content)
                        if crc_final == int(crc_data, 16):
                            # 数据校验成功，保存数据
                            file.write(content)
                            # number = int(s.read().hex(), 16)  # 计数号码
                            number = s.read()  # 读取计数号码，但不处理
                            total += 1
                        else:
                            # 校验失败
                            pass
                    else:
                        pass
                else:
                    pass
            else:
                print("第一个字节校验失败")
                continue


def draw(data):
    plt.figure(figsize=(10, 8))
    plt.subplot(311)
    plt.plot(data['ir2'])
    plt.title('ir2')
    plt.subplot(312)
    plt.plot(data['red2'])
    plt.title('red2')
    plt.subplot(313)
    plt.plot(data['resp'])
    plt.title("resp")
    plt.subplots_adjust(hspace=0.8)
    plt.show()


def collect_test():
    path = r"./read_test_data.txt"
    read_package(path, num=30000)
    data = pd.read_table(path, sep=",")
    if data.shape[1] == 6:
        data.columns = ["time", "ir1", "red1", "ir2", "red2", "resp"]
    elif data.shape[1] == 5:
        data.columns = ["ir1", "red1", "ir2", "red2", "resp"]
    else:
        data.columns = ["ir1", "red1", "ir2", "red2"]
    draw(data)
    os.remove("./read_test_data.txt")


if __name__ == '__main__':
    collect_test()
