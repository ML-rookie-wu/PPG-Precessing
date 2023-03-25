# -*- coding: utf-8 -*-

"""
@author: Mark Wu
@file: whale.py
@time: 2023/3/15 16:41
"""

import numpy as np
from scipy.signal import chirp, find_peaks, peak_widths
from scipy.io import loadmat
import matplotlib.pyplot as plt
from scipy.signal import hilbert
from codes.utils.GetFileData import read_from_file


class VMD:
    def __init__(self, K, alpha, tau, tol=1e-7, maxIters=200, eps=1e-9):
        """
        :param K: 模态数
        :param alpha: 每个模态初始中心约束强度
        :param tau: 对偶项的梯度下降学习率
        :param tol: 终止阈值
        :param maxIters: 最大迭代次数
        :param eps: eps
        """
        self.K = K
        self.alpha = alpha
        self.tau = tau
        self.tol = tol
        self.maxIters = maxIters
        self.eps = eps

    def __call__(self, f):
        N = f.shape[0]
        # 对称拼接
        f = np.concatenate((f[:N // 2][::-1], f, f[N // 2:][::-1]))
        T = f.shape[0]
        t = np.linspace(1, T, T) / T
        omega = t - 1. / T
        # 转换为解析信号
        f = hilbert(f)
        f_hat = np.fft.fft(f)
        u_hat = np.zeros((self.K, T), dtype=np.complex)
        omega_K = np.zeros((self.K,))
        lambda_hat = np.zeros((T,), dtype=np.complex)
        # 用以判断
        u_hat_pre = np.zeros((self.K, T), dtype=np.complex)
        u_D = self.tol + self.eps

        # 迭代
        n = 0
        while n < self.maxIters and u_D > self.tol:
            for k in range(self.K):
                # u_hat
                sum_u_hat = np.sum(u_hat, axis=0) - u_hat[k, :]
                res = f_hat - sum_u_hat
                u_hat[k, :] = (res + lambda_hat / 2) / (1 + self.alpha * (omega - omega_K[k]) ** 2)

                # omega
                u_hat_k_2 = np.abs(u_hat[k, :]) ** 2
                omega_K[k] = np.sum(omega * u_hat_k_2) / np.sum(u_hat_k_2)

            # lambda_hat
            sum_u_hat = np.sum(u_hat, axis=0)
            res = f_hat - sum_u_hat
            lambda_hat -= self.tau * res

            n += 1
            u_D = np.sum(np.abs(u_hat - u_hat_pre) ** 2)
            u_hat_pre[::] = u_hat[::]

        # 重构，反傅立叶之后取实部
        u = np.real(np.fft.ifft(u_hat, axis=-1))
        u = u[:, N // 2: N // 2 + N]

        omega_K = omega_K * T / 2
        idx = np.argsort(omega_K)

        omega_K = omega_K[idx]
        u = u[idx, :]
        return u, omega_K


# 定义信号重构误差作为优化目标
def cost_func(x, data):
    tau = 1e-6
    # x为待优化参数，s为原始信号
    vmd = VMD(x[0], x[1], tau)
    data = np.array(data)
    u, omega_K = vmd(data)
    restruct = np.sum(u, 0)
    arr_data = np.array(data)
    # 使用重构的信号与原始信号计算均方误差
    error = np.mean(abs(restruct - arr_data))
    print("error = ", error)
    return error


# 定义鲸鱼算法
def whale_algorithm(data, cost_func, lb, ub, dim, max_iter, pop_size):
    # 初始化种群
    X = np.random.randint(lb, ub, (pop_size, dim))
    # print(X)
    # 初始化个体最优位置和适应度
    P_best = X.copy()
    P_best_fitness = np.zeros(pop_size)
    for i in range(pop_size):
        print("i = ", i)
        print(X[i])
        P_best_fitness[i] = cost_func(X[i], data)
    print(P_best_fitness)
    # 寻找全局最优位置和适应度
    gbest_idx = np.argmin(P_best_fitness)
    gbest = P_best[gbest_idx].copy()
    gbest_fitness = P_best_fitness[gbest_idx]
    # 开始迭代
    for t in range(max_iter):
        # 计算a和A
        a = 2 - 2 * t / max_iter
        A = 2 * a * np.random.rand(pop_size, dim) - a
        # 计算C和p
        p = np.random.rand(pop_size, dim)
        C = 1 / (1 + np.exp(-A))
        # 更新位置和个体最优位置
        X = X + np.multiply(C, np.subtract(P_best, X)) + np.multiply(p, np.subtract(gbest, X))
        print("X = ", X)
        # 处理越界的位置
        X[X < lb] = lb
        X[X > ub] = ub
        # 计算适应度
        fitness = np.zeros(pop_size)
        for i in range(pop_size):
            fitness[i] = cost_func(X[i], data)
            print("fitness---", fitness)
            # 更新个体最优位置和全局最优位置
            if fitness[i] < P_best_fitness[i]:
                P_best[i] = X[i].copy()
                P_best_fitness[i] = fitness[i]
                if P_best_fitness[i] < gbest_fitness:
                    gbest = P_best[i].copy()
                    gbest_fitness = P_best_fitness[i]
        # 输出当前迭代的结果
        print('Iteration {}: Best fitness = {:.6f}'.format(t+1, gbest_fitness))
    return gbest, gbest_fitness


def main():
    file_path = r"D:\my_projects_V1\my_projects\PPG_V1\data\respiration\2\20230314200844.txt"
    data = read_from_file(file_path)
    ir = data.ir2
    raw_data = ir[5000:9000]
    popsize = 5
    iter = 20
    dim = 2
    lb = [5000, 3]
    ub = [6000, 15]
    gbest, gbest_fitness = whale_algorithm(raw_data, cost_func, lb, ub, dim, iter, popsize)


if __name__ == '__main__':
    main()