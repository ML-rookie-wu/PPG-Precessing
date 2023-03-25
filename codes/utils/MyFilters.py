#coding:utf-8

import pywt
from PyEMD import EMD, Visualisation
import numpy as np
import glob
import scipy.signal as scisignal
import math
from scipy.signal import hilbert


def reverse(data):
    """
    反转波形
    """
    data_max = max(data)

    reversed_data = [data_max - _ for _ in data]

    return reversed_data

def bandpass_filter(data, fs=500, start_fs=0.1, end_fs=0.8):
    """巴特沃兹带通滤波"""
    winHz = [start_fs, end_fs]
    wn1 = 2 * winHz[0] / fs
    wn2 = 2 * winHz[1] / fs
    b, a = scisignal.butter(2, [wn1, wn2],
                            'bandpass')  # 截取频率[1，5]hz #这里假设采样频率为1000hz,信号本身最大的频率为500hz，要滤除10hz以下和400hz以上频率成分，即截至频率为10hz和400hz,则wn1=2*10/1000=0.02,wn2=2*400/1000=0.86。Wn=[0.02,0.86]#https://blog.csdn.net/weixin_37996604/article/details/82864680
    data = scisignal.filtfilt(b, a, data)  # data为要过滤的信号
    return data

def kalman_filter(data, q=0.0001, r=0.01):
    # 后验初始值
    x0 = data[0]                              # 令第一个估计值，为当前值
    p0 = 1.0
    # 存结果的列表
    x = [x0]
    for z in data[1:]:                        # kalman 滤波实时计算，只要知道当前值z就能计算出估计值(后验值)x0
        # 先验值
        x1_minus = x0                         # X(k|k-1) = AX(k-1|k-1) + BU(k) + W(k), A=1,BU(k) = 0
        p1_minus = p0 + q                     # P(k|k-1) = AP(k-1|k-1)A' + Q(k), A=1
        # 更新K和后验值
        k1 = p1_minus / (p1_minus + r)        # Kg(k)=P(k|k-1)H'/[HP(k|k-1)H' + R], H=1
        x0 = x1_minus + k1 * (z - x1_minus)   # X(k|k) = X(k|k-1) + Kg(k)[Z(k) - HX(k|k-1)], H=1
        p0 = (1 - k1) * p1_minus              # P(k|k) = (1 - Kg(k)H)P(k|k-1), H=1
        x.append(x0)                          # 由输入的当前值z 得到估计值x0存入列表中，并开始循环到下一个值
    return x

def get_dwt_res(data, w='db33', n=10):
    """Decompose and plot a signal S.
    S = An + Dn + Dn-1 + ... + D1
    """
    mode = pywt.Modes.smooth
    w = pywt.Wavelet(w)            # 选取小波函数
    a = data
    ca = []   # 近似分量, a表示低频近似部分
    cd = []   # 细节分量, b表示高频细节部分
    for i in range(n):
        (a, d) = pywt.dwt(a, w, mode)#进行n阶离散小波变换
        ca.append(a)
        cd.append(d)

    rec_a = []
    rec_d = []

    for i, coeff in enumerate(ca):
        # print("cai-----", len(ca[i]))
        coeff_list = [coeff, None] + [None] * i
        rec_a.append(pywt.waverec(coeff_list, w))      #重构

    for i, coeff in enumerate(cd):  # i, coeff 分别对应ca中的下标和元素，分了几层i就为几
        coeff_list = [None, coeff] + [None] * i
        rec_d.append(pywt.waverec(coeff_list, w))

    return rec_a, rec_d

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

def vmd(data):
    K = 8
    alpha = 5000
    tau = 1e-6
    vmd = VMD(K, alpha, tau)
    u, omega_K = vmd(data)
    results = data - u[0]
    return results, u, u[0]

def dwt(data):
    buttered = bandpass_filter(data, start_fs=0.1, end_fs=5)

    ac, dc = get_dwt_res(buttered)
    resp_disturb = ac[-2][0:len(data)]
    removed = [buttered[i] - resp_disturb[i] for i in range(len(data))]

    return removed


def MyProcess(filter_name: str):
    if filter_name == "vmd":
        return vmd
    elif filter_name == "dwt":
        return dwt
