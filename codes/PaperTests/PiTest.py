# -*- coding: utf-8 -*-

"""
@author: Mark Wu
@file: PiTest.py
@time: 2023/1/17 10:48
"""

import numpy as np
import math
import matplotlib.pyplot as plt
from codes.utils.GetFileData import read_from_file
from codes.utils.MyFilters import reverse, bandpass_filter, dwt

from codes.utils.Normal import normalization


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

def PI(data):
    # data = normalization(data)
    dc = np.mean(data)
    # ac = sum([(x - dc) for x in data])
    # pi = (ac / dc) * 100
    # print("pi =", pi)

    filtered_data = dwt(data)
    ac_list = [(x - dc) for x in data]
    ac = math.sqrt(sum(x ** 2 for x in ac_list) / len(data))
    ac1 = math.sqrt(sum(x ** 2 for x in filtered_data) / len(data))
    pi = (ac / dc) * 100
    pi1 = (ac1 / dc) * 100
    print("ac = %s, dc = %s, ac1 = %s" % (ac, dc, ac1))
    print("pi = %s, pi1 = %s" % (pi, pi1))
    # plt.subplot(211)
    # plt.plot(ir2)
    # plt.subplot(212)
    # plt.plot(but_ir2)
    # plt.show()

if __name__ == '__main__':
    path = r'D:\my_projects_V1\my_projects\PPG_V1\data\spo2_compare_new\0disturb_3pulse\98\20221130184623_98.txt'
    ir2, red2 = read_data(path, label="simulator")
    PI(ir2)
