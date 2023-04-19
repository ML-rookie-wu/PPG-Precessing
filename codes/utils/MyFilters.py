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


import numpy as np


def VMD1(f, alpha, tau, K, tol=1e-7, DC=0, init=1):
    """
    u,u_hat,omega = VMD(f, alpha, tau, K, DC, init, tol)
    Variational mode decomposition
    Python implementation by Vinícius Rezende Carvalho - vrcarva@gmail.com
    code based on Dominique Zosso's MATLAB code, available at:
    https://www.mathworks.com/matlabcentral/fileexchange/44765-variational-mode-decomposition
    Original paper:
    Dragomiretskiy, K. and Zosso, D. (2014) ‘Variational Mode Decomposition’,
    IEEE Transactions on Signal Processing, 62(3), pp. 531–544. doi: 10.1109/TSP.2013.2288675.


    Input and Parameters:
    ---------------------
    f       - the time domain signal (1D) to be decomposed
    alpha   - the balancing parameter of the data-fidelity constraint
    tau     - time-step of the dual ascent ( pick 0 for noise-slack )
    K       - the number of modes to be recovered
    DC      - true if the first mode is put and kept at DC (0-freq)
    init    - 0 = all omegas start at 0
                       1 = all omegas start uniformly distributed
                      2 = all omegas initialized randomly
    tol     - tolerance of convergence criterion; typically around 1e-6
    Output:
    -------
    u       - the collection of decomposed modes
    u_hat   - spectra of the modes
    omega   - estimated mode center-frequencies
    """

    if len(f) % 2:
        f = f[:-1]

    # Period and sampling frequency of input signal
    fs = 1. / len(f)

    ltemp = len(f) // 2
    fMirr = np.append(np.flip(f[:ltemp], axis=0), f)
    fMirr = np.append(fMirr, np.flip(f[-ltemp:], axis=0))

    # Time Domain 0 to T (of mirrored signal)
    T = len(fMirr)
    t = np.arange(1, T + 1) / T

    # Spectral Domain discretization
    freqs = t - 0.5 - (1 / T)

    # Maximum number of iterations (if not converged yet, then it won't anyway)
    Niter = 500
    # For future generalizations: individual alpha for each mode
    Alpha = alpha * np.ones(K)

    # Construct and center f_hat
    f_hat = np.fft.fftshift((np.fft.fft(fMirr)))
    f_hat_plus = np.copy(f_hat)  # copy f_hat
    f_hat_plus[:T // 2] = 0

    # Initialization of omega_k
    omega_plus = np.zeros([Niter, K])

    if init == 1:
        for i in range(K):
            omega_plus[0, i] = (0.5 / K) * (i)
    elif init == 2:
        omega_plus[0, :] = np.sort(np.exp(np.log(fs) + (np.log(0.5) - np.log(fs)) * np.random.rand(1, K)))
    else:
        omega_plus[0, :] = 0

    # if DC mode imposed, set its omega to 0
    if DC:
        omega_plus[0, 0] = 0

    # start with empty dual variables
    lambda_hat = np.zeros([Niter, len(freqs)], dtype=complex)

    # other inits
    uDiff = tol + np.spacing(1)  # update step
    n = 0  # loop counter
    sum_uk = 0  # accumulator
    # matrix keeping track of every iterant // could be discarded for mem
    u_hat_plus = np.zeros([Niter, len(freqs), K], dtype=complex)

    # *** Main loop for iterative updates***

    while (uDiff > tol and n < Niter - 1):  # not converged and below iterations limit
        # update first mode accumulator
        k = 0
        sum_uk = u_hat_plus[n, :, K - 1] + sum_uk - u_hat_plus[n, :, 0]

        # update spectrum of first mode through Wiener filter of residuals
        u_hat_plus[n + 1, :, k] = (f_hat_plus - sum_uk - lambda_hat[n, :] / 2) / (
                    1. + Alpha[k] * (freqs - omega_plus[n, k]) ** 2)

        # update first omega if not held at 0
        if not (DC):
            omega_plus[n + 1, k] = np.dot(freqs[T // 2:T], (abs(u_hat_plus[n + 1, T // 2:T, k]) ** 2)) / np.sum(
                abs(u_hat_plus[n + 1, T // 2:T, k]) ** 2)

        # update of any other mode
        for k in np.arange(1, K):
            # accumulator
            sum_uk = u_hat_plus[n + 1, :, k - 1] + sum_uk - u_hat_plus[n, :, k]
            # mode spectrum
            u_hat_plus[n + 1, :, k] = (f_hat_plus - sum_uk - lambda_hat[n, :] / 2) / (
                        1 + Alpha[k] * (freqs - omega_plus[n, k]) ** 2)
            # center frequencies
            omega_plus[n + 1, k] = np.dot(freqs[T // 2:T], (abs(u_hat_plus[n + 1, T // 2:T, k]) ** 2)) / np.sum(
                abs(u_hat_plus[n + 1, T // 2:T, k]) ** 2)

        # Dual ascent
        lambda_hat[n + 1, :] = lambda_hat[n, :] + tau * (np.sum(u_hat_plus[n + 1, :, :], axis=1) - f_hat_plus)

        # loop counter
        n = n + 1

        # converged yet?
        uDiff = np.spacing(1)
        for i in range(K):
            uDiff = uDiff + (1 / T) * np.dot((u_hat_plus[n, :, i] - u_hat_plus[n - 1, :, i]),
                                             np.conj((u_hat_plus[n, :, i] - u_hat_plus[n - 1, :, i])))

        uDiff = np.abs(uDiff)

        # Postprocessing and cleanup

    # discard empty space if converged early
    Niter = np.min([Niter, n])
    omega = omega_plus[:Niter, :]

    idxs = np.flip(np.arange(1, T // 2 + 1), axis=0)
    # Signal reconstruction
    u_hat = np.zeros([T, K], dtype=complex)
    u_hat[T // 2:T, :] = u_hat_plus[Niter - 1, T // 2:T, :]
    u_hat[idxs, :] = np.conj(u_hat_plus[Niter - 1, T // 2:T, :])
    u_hat[0, :] = np.conj(u_hat[-1, :])

    u = np.zeros([K, len(t)])
    for k in range(K):
        u[k, :] = np.real(np.fft.ifft(np.fft.ifftshift(u_hat[:, k])))

    # remove mirror part
    u = u[:, T // 4:3 * T // 4]

    # recompute spectrum
    u_hat = np.zeros([u.shape[1], K], dtype=complex)
    for k in range(K):
        u_hat[:, k] = np.fft.fftshift(np.fft.fft(u[k, :]))

    return u, u_hat, omega

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

def vmd(data, K=8, alpha=5000):
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
