#coding:utf-8

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert
import scipy.signal as scisignal
from codes.utils.GetFileData import read_from_file
from codes.utils.GetFFT import signal_fft, get_freq



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

def bandpass_filter(data, fs=500, start_fs=0.1, end_fs=0.8):
    winHz = [start_fs, end_fs]
    wn1 = 2 * winHz[0] / fs
    wn2 = 2 * winHz[1] / fs
    b, a = scisignal.butter(2, [wn1, wn2],
                            'bandpass')  # 截取频率[1，5]hz #这里假设采样频率为1000hz,信号本身最大的频率为500hz，要滤除10hz以下和400hz以上频率成分，即截至频率为10hz和400hz,则wn1=2*10/1000=0.02,wn2=2*400/1000=0.86。Wn=[0.02,0.86]#https://blog.csdn.net/weixin_37996604/article/details/82864680
    data = scisignal.filtfilt(b, a, data)  # data为要过滤的信号
    return data

def cal_vmd(ppg_data):
    K = 8
    alpha = 5000
    tau = 1e-6
    vmd = VMD(K, alpha, tau)
    ppg_data = np.array(ppg_data)
    u, omega_K = vmd(ppg_data)

    vmd_filed = ppg_data - u[0]
    resp = u[0]
    fil_resp = bandpass_filter(resp, start_fs=0.1, end_fs=0.8)
    current = resp - fil_resp

    return resp, vmd_filed, current


def plot_vmd(ppg_data):
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

    # splited_ppg, splited_resp = split_data(ppg_data, resp_data, 0, 25000)
    # splited_ppg = splited_ppg.reindex

    K = 8
    alpha = 5000
    tau = 1e-6
    vmd = VMD(K, alpha, tau)
    u, omega_K = vmd(ppg_data)

    # omega_K
    # array([  9.68579292,  50.05232833, 100.12321047])

    # vmd分解绘图，每一层
    # plt.figure(figsize=(10, 86))
    # plt.subplot(len(u) + 1, 1, 1)
    # plt.plot(ppg_data)
    # plt.subplots_adjust(hspace=0.86)
    #
    # for i in range(len(u)):
    #     plt.subplot(len(u) + 1, 2, 3 + i * 2)
    #     plt.plot(u[i])
    #     plt.title('%s层分量' % i)
    #     plt.subplots_adjust(hspace=1.2)
    #
    # for i in range(len(u)):
    #     f, absY = signal_fft(u[i], 100)
    #     plt.subplot(len(u) + 1, 2, 4 + i * 2)
    #     plt.plot(f, absY)
    #     plt.title('%s层分量' % i)
    #     plt.subplots_adjust(hspace=0.86)

    # plt.show()

    # results = ppg_data - u[0]
    results = ppg_data - u[0]

    resp = u[0]
    fil_data = bandpass_filter(results, start_fs=1, end_fs=5)
    fil_resp = bandpass_filter(resp, start_fs=0.1, end_fs=0.8)
    freq, amplitude = signal_fft(resp, 500)
    # get_freq(freq, amplitude)
    raw_ppg_freq, raw_ppg_amp = signal_fft(ppg_data, 500)

    plt.figure(figsize=(10, 8))
    plt.subplot(411)
    plt.plot(ppg_data)
    plt.plot(u[0], c='r')
    plt.title('raw_ppg')
    plt.subplot(412)
    plt.plot(results)
    plt.title('filted')
    plt.subplot(413)
    plt.plot(raw_ppg_freq, raw_ppg_amp, color='r')
    plt.title("respiratory")
    plt.subplot(414)
    plt.plot(freq, amplitude)
    plt.title("frequency")
    plt.subplots_adjust(hspace=0.8)
    plt.show()

    return resp

def main():
    path = r'D:\my_projects_V1\my_projects\pyqt\learnDemo\data\2022_5_3_15_48.txt'
    # path = r'D:\my_projects_V1\my_projects\PPG_V1\data\spo2_compare_test\1disturb_3pulse\100\20221129193845_100.txt'
    data = read_from_file(path)
    ir2 = data.ir2
    buttered = bandpass_filter(ir2, start_fs=0.1, end_fs=5)
    partial = ir2[2000:10000]
    plot_vmd(buttered)

if __name__ == '__main__':
    main()
