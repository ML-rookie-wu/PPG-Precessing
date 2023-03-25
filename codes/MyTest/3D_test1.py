# -*- coding: utf-8 -*-

"""
@author: Mark Wu
@file: 3D_test1.py
@time: 2023/2/23 11:00
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Mar 26 17:39:30 2022

@author: keyanxiaobai
"""

import matplotlib.pyplot as plt
import numpy as np
from codes.utils.MyFilters import vmd
from codes.utils.GetFileData import read_from_file
# from vmdpy import VMD
import pandas as pd
# import torch
import scipy.io as scio

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# . some sample parameters for VMD
alpha = 2000  # moderate bandwidth constraint
tau = 0.  # noise-tolerance (no strict fidelity enforcement)
K = 5  # 5 modes
DC = 0  # no DC part imposed
init = 1  # initialize omegas uniformly
tol = 1e-7


path = r'D:\my_projects_V1\my_projects\PPG_V1\data\paper_test\5disturb_3pulse\98\20221130161323_98.txt'
data = read_from_file(path).ir2
# data为你要分解的信号
# . Run VMD
# u, u_hat, omega = VM(data, alpha, tau, K, DC, init, tol)
results, u, resp = vmd(data)
print(len(u))
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

# colors = ['r', 'g', 'b', 'y']
# yticks = [3, 2, 1, 0]
colors = ['r', 'g', 'b', 'm', 'c', 'pink', 'purple', 'cyan']
yticks = list(range(len(u)))
# datas=[data,data1]
# datass = [u[0], u[1], u[2], u[3], u[4]]
for c, k, d in zip(colors, yticks, u):
    # Generate the random data for the y=k 'layer'.
    # xs = np.arange(1, 1001)
    xs = np.arange(0, len(data))
    print(c)
    print(k)
    print(d)

    # Plot the bar graph given by xs and ys on the plane y=k with 80% opacity.
    ax.plot(xs, d, zs=k, zdir='y', color=c, alpha=0.8)

ax.set_xlabel('序列长度')
ax.set_ylabel('模态分量/（个）')
ax.set_zlabel('幅值')

# On the y axis let's only label the discrete values that we have data for.
ax.set_yticks(yticks)
# plt.savefig('./vmd_3D.svg', dpi=400,bbox_inches='tight', pad_inches=0)  # dpi决定保存的图片的清晰度
plt.show()
