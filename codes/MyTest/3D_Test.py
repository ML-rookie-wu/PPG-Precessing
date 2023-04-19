# -*- coding: utf-8 -*-

"""
@author: Mark Wu
@file: 3D_Test.py
@time: 2023/2/23 10:48
"""

import numpy as np
import os
import matplotlib.pyplot as plt
from codes.utils.GetFileData import read_from_file
from codes.utils.MyFilters import vmd, bandpass_filter
from random import sample
from codes.utils.GetFFT import signal_fft
from codes.PaperTests import PAPER_FIGURE_PATH

plt.rcParams['font.sans-serif'] = ['SongNTR']
plt.rcParams['axes.unicode_minus'] = False

# fig = plt.figure()
# ax3d = fig.add_subplot(projection='3d')  #创建3d坐标系
#
# x,y=np.mgrid[-3:3:0.2,-3:3:0.2]
# z=x*np.exp(-x**2-y**2)
#
# #ax3d.contour(x,y,z)
# # ax3d.contour(x,y,z,levels=10,cmap="coolwarm")  #指定等高线数和颜色
# # ax3d.contourf(x,y,z,levels=10,cmap="coolwarm") #填充等高线
# ax3d.contour(x,y,z,zdir='x',levels=10)  #x方向等高线

#投影
#ax3d.contour(x,y,z,levels=10,zdir='x',offset=-3)
#ax3d.contour(x,y,z,levels=10,zdir='y',offset=3)
#ax3d.contour(x,y,z,levels=10,zdir='z',offset=-0.4)

# plt.show()

def normal(data):
    data_max = np.max(data)
    data_min = np.min(data)
    normaled = [(x - data_min) / (data_max - data_min) for x in data]
    return normaled

def test():
    path = r'D:\my_projects_V1\my_projects\PPG_V1\data\paper_test\5disturb_3pulse\98\20221130161323_98.txt'
    data = read_from_file(path).ir2[0:4000]
    result, u, resp = vmd(data)
    u = np.array(u)
    print(u.shape)
    x = list(range(len(data)))
    y = np.arange(0, len(u)*10, len(u))
    color_list = ["gold", "springgreen", "turquoise", "cyan", "dodgerblue", "darkviolet", "deeppink", "darkmagenta", "darkorange", "tomato"]
    c = sample(color_list, len(u))
    fig = plt.figure(figsize=(16, 10))
    ax3d = fig.add_subplot(projection='3d')  # 创建3d坐标系
    # fig1 = plt.figure(figsize=(10, 8))
    # ax3d1 = fig1.add_subplot(projection='3d')  # 创建3d坐标系
    # ax3d.set_xticks(np.arange(0, len(u)*10, 10))

    for i in range(len(u)):
        z = normal(u[i])
        ax3d.plot(x, z, i, zdir="x", color=c[i])
        # if i == 0:
        #     u[i] = bandpass_filter(u[i], start_fs=0.1, end_fs=0.8)
        # f, absY = signal_fft(u[i])
        # ax3d1.plot(f, absY, i, zdir="x", color=c[i])
    # ax3d.set_xlim(0, 3)
    # ax3d.set_xlabel(["模态1", "模态2", "模态3", "模态4", "模态5", "模态6", "模态7", "模态8"])
    # ax3d.set_xticks(range())
    ax3d.set_xlabel("模态分量")
    ax3d.set_ylabel("采样点", labelpad=10)
    ax3d.set_title("VMD分解")
    plt.savefig(os.path.join(PAPER_FIGURE_PATH, "vmd3d分解"), dpi=300, bbox_inches="tight")
    plt.show()

    fig1 = plt.figure(figsize=(10, 8))
    ax3d1 = fig1.add_subplot(projection='3d')  # 创建3d坐标系
    for i in range(len(u)):
        if i == 0:
            u[i] = bandpass_filter(u[i], start_fs=0.1, end_fs=0.8)
        f, absY = signal_fft(u[i])
        ax3d1.plot(f, absY, i, zdir="x", color=c[i])
    ax3d1.set_xticks(range(len(u)))
    ax3d1.set_xlabel("模态分量")
    ax3d1.set_ylabel("频率", labelpad=10)
    ax3d1.set_title("模态分量频谱")
    plt.savefig(os.path.join(PAPER_FIGURE_PATH, "vmd3d频谱"), dpi=300, bbox_inches="tight")
    plt.show()


    # for i in range(len(u)):
    #     # z = normal(u[i])
    #     f, amp = signal_fft(u[i])
    #     # z = u[i]
    #     ax3d1.plot(x, z, i, zdir="x", color=c[i])
    # # ax3d.set_xlim(0, 3)
    # # ax3d.set_xticks(["模态1", "模态2", "模态3", "模态4"])
    # ax3d.set_xticks(range(len(u)))
    #
    # ax3d.set_xlabel("模态分量")
    # ax3d.set_ylabel("采样点", labelpad=10)
    # ax3d.set_title("VMD分解")
    # # ax3d.plot(x, y, u, zdir="y")
    # plt.show()

def test2():
    import matplotlib as mpl
    from mpl_toolkits.mplot3d import Axes3D
    import numpy as np
    import matplotlib.pyplot as plt

    x = np.arange(8)
    y = np.random.randint(0, 10, 8)

    y2 = y + np.random.randint(0, 3, 8)
    y3 = y2 + np.random.randint(0, 3, 8)
    y4 = y3 + np.random.randint(0, 3, 8)
    y5 = y4 + np.random.randint(0, 3, 8)
    print(y2)

    clr = ['red', 'green', 'blue', 'black', 'white', 'yellow', 'orange', 'pink']

    fig = plt.figure()
    ax = Axes3D(fig)

    ax.bar(x, y, 0, zdir='y', color=clr)
    ax.bar(x, y2, 10, zdir='y', color=clr)
    ax.bar(x, y3, 20, zdir='y', color=clr)
    ax.bar(x, y4, 30, zdir='y', color=clr)
    ax.bar(x, y5, 40, zdir='y', color=clr)

    ax.set_xlabel('X label')
    ax.set_ylabel('Y label')
    ax.set_zlabel('Z label')

    plt.show()

test()
# test2()

a = [[1, 2, 3], [4, 5, 6]]
b = np.array(a)
# print(b.shape)