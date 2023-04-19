# -*- coding: utf-8 -*-

"""
@author: Mark Wu
@file: SimulatorPlot.py
@time: 2023/2/12 15:13
"""
import matplotlib.pyplot as plt
import os
from codes.PaperTests import PAPER_FIGURE_PATH
from codes.utils.GetFileData import read_from_file


plt.rcParams["font.sans-serif"]=["SongNTR"] #设置字体
plt.rcParams["axes.unicode_minus"]=False #正常显示负号


def no_resp_disturb():
    path = r'D:\my_projects_V1\my_projects\PPG_V1\data\spo2_compare_new\0disturb_3pulse\98\20221130184623_98.txt'
    data = read_from_file(path)
    ir2 = data.ir2
    plt.figure(figsize=(10, 8))
    # fig, ax = plt.subplots()   # 这样创建画布，可以是图片小一点
    plt.plot(ir2[0:5000], label="PPG")
    plt.title("无呼吸干扰的PPG")
    plt.ylabel("幅值")
    plt.xlabel("采样点")
    # ax.spines["top"].set_visible(False)
    # ax.spines["right"].set_visible(False)
    plt.legend(loc="best")
    plt.savefig(os.path.join(PAPER_FIGURE_PATH, "无呼吸模拟仪数据"), dpi=300, bbox_inches="tight")
    plt.show()

def resp_disturb():
    path = r'D:\my_projects_V1\my_projects\PPG_V1\data\spo2_compare_new\3disturb_3pulse\98\20221130141715_98.txt'
    data = read_from_file(path)
    ir2 = data.ir2
    plt.figure(figsize=(10, 8))
    plt.plot(ir2[0:5000], label="PPG")
    plt.title("有呼吸干扰的PPG")
    plt.ylabel("幅值")
    plt.xlabel("采样点")
    plt.legend(loc="best")
    plt.savefig(os.path.join(PAPER_FIGURE_PATH, "有呼吸模拟仪数据"), dpi=300, bbox_inches="tight")
    plt.show()

def main():
    no_resp_disturb()
    resp_disturb()


if __name__ == '__main__':
    main()