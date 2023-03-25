# -*- coding: utf-8 -*-

"""
@author: Mark Wu
@file: VMD_piles.py
@time: 2023/3/16 20:54
"""
import numpy as np
import matplotlib.pyplot as plt
from codes.utils.MyFilters import VMD
from codes.utils.GetFileData import read_from_file



def cost_func(restruct, data):
    arr_data = np.array(data)
    # 使用重构的信号与原始信号计算均方误差
    error = np.mean((restruct - arr_data)**2)
    print("error = ", error)
    return error

def loss_mape(u, data):
    total = 0
    for i in range(len(data)):
        u_k = np.sum([x[i] for x in u])
        # print(u_k, data[i])
        total += abs((data[i] - u_k) / data[i])
    return total / len(data)

def main():
    file_path = r"D:\my_projects_V1\my_projects\PPG_V1\data\respiration\2\20230314200844.txt"
    data = read_from_file(file_path)
    ir = data.ir2
    raw_data = np.array(ir[5000:9000])

    # K = 8
    alpha = 5000
    tau = 1e-6
    Klist = range(3, 15)
    Elist = []
    for K in Klist:
        vmd = VMD(K, alpha, tau)
        u, omega = vmd(raw_data)
        restruct = np.sum(u, 0)
        print("k = ", K)
        print(omega)
        # mape_loss = loss_mape(u, raw_data)
        # print(mape_loss)
        # E_k = 0
        # for u_i in u:
        #     e_i = np.sqrt(np.mean([x ** 2 for x in u_i]))
        #     E_k += e_i
        # Elist.append(E_k)
        # print("k = ", K)
        # print("E_k = ", E_k)
        # error = cost_func(restruct, raw_data)
    # plt.plot(Elist)
    # plt.show()


if __name__ == '__main__':
    main()
