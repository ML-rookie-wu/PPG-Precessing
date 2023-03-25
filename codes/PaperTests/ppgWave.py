# -*- coding: utf-8 -*-

"""
@author: Mark Wu
@file: ppgWave.py
@time: 2023/2/7 10:07
"""


import matplotlib.pyplot as plt
import os
from codes.utils.GetFileData import read_from_file
from codes.utils.MyFilters import bandpass_filter
from codes.PaperTests import PAPER_FIGURE_PATH

def reverse(data):
    """
    反转波形
    """
    data_max = max(data)
    reversed_data = [data_max - _ for _ in data]
    return reversed_data

def read_data():
    path = r'D:\my_projects_V1\my_projects\PPG_V1\data\spo2_compare_new\0disturb_3pulse\97\20221130185157_97.txt'
    data = read_from_file(path)
    ir2 = data.ir2
    reverse_ir2 = reverse(ir2)
    filtered_ir2 = bandpass_filter(reverse_ir2, start_fs=0.1, end_fs=5)
    return filtered_ir2

def plot_data(data, save=False):
    pic_name = "ppg_wave"
    pic_path = os.path.join(PAPER_FIGURE_PATH, pic_name)
    plt.figure(figsize=(10, 8))
    plt.plot(data[9700:10300])
    # plt.tick_params(axis='both', which='both')  # 刻度线
    plt.axis('off')    # 去掉边框
    if save:
        plt.savefig(pic_path, dpi=200)
    plt.show()

def main():
    ir2 = read_data()
    plot_data(ir2, save=True)


if __name__ == '__main__':
    main()
