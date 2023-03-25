# -*- coding: utf-8 -*-

"""
@author: Mark Wu
@file: LMSTest1.py
@time: 2023/1/27 11:21
"""



import numpy as np

import matplotlib.pyplot as plt
import random
import scipy as sc
import pandas as pd
from codes.utils.GetFileData import read_from_file


# 定义向量的内积

def multiVector(A, B):
    C = sc.zeros(len(A))

    for i in range(len(A)):
        C[i] = A[i] * B[i]

    return sum(C)


# 取定给定的反向的个数
def inVector(A, b, a):
    D = sc.zeros(b - a + 1)
    for i in range(b - a + 1):
        D[i] = A[i + a]
    return D[::-1]


# lMS算法的函数
def LMS(xn, dn, M, mu, itr):
    en = sc.zeros(itr)
    W = [[0] * M for i in range(itr)]
    for k in range(itr)[M - 1:itr]:
        x = inVector(xn, k, k - M + 1)
        d = x.mean()
        y = multiVector(W[k - 1], x)
        en[k] = d - y
        W[k] = np.add(W[k - 1], 2 * mu * en[k] * x)  # 跟新权重

    # 求最优时滤波器的输出序列
    yn = sc.inf * sc.ones(len(xn))
    for k in range(len(xn))[M - 1:len(xn)]:
        x = inVector(xn, k, k - M + 1)
        yn[k] = multiVector(W[len(W) - 1], x)

    return (yn, en)


if __name__ == "__main__":

    path = r'D:\my_projects_V1\my_projects\PPG_V1\data\spo2_compare_new\3disturb_3pulse\98\20221130141715_98.txt'
    data = read_from_file(path)
    ppg_data = data.ir2

    # 参数初始化

    itr = len(ppg_data)  # 采样的点数
    mu = 0
    sigma = 0.12

    X = np.linspace(0, 4 * np.pi, itr, endpoint=True)
    Y = np.sin(X)
    signal_array = Y  # [0.0]*noise_size


    M = 64  # 滤波器的阶数

    mu = 0.0001  # 步长因子

    xs = ppg_data

    xn = xs  # 原始输入端的信号为被噪声污染的正弦信号

    dn = ppg_data  # 对于自适应对消器，用dn作为期望

    # 调用LMS算法

    (yn, en) = LMS(xn, dn, M, mu, itr)

    plt.figure(figsize=(10, 8))
    plt.subplot(211)
    plt.plot(ppg_data)
    plt.subplot(212)
    plt.plot(yn)
    plt.show()
