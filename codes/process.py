#coding:utf-8
import os

import pywt
import matplotlib.pyplot as plt
from PyEMD import EMD, Visualisation
import numpy as np
from scipy.signal import hilbert
import pandas as pd
import glob
import scipy.signal as scisignal
import math
from scipy import signal
import openpyxl


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
        coeff_list = [coeff, None] + [None] * i
        rec_a.append(pywt.waverec(coeff_list, w))      #重构

    for i, coeff in enumerate(cd):  # i, coeff 分别对应ca中的下标和元素，分了几层i就为几
        coeff_list = [None, coeff] + [None] * i
        rec_d.append(pywt.waverec(coeff_list, w))
    # print(rec_a, type(rec_a))
    # print(rec_d, type(rec_d))

    fig = plt.figure(figsize=(16, 12))
    ax_main = fig.add_subplot(len(rec_a) + 1, 1, 1)
    # ax_main.set_title(title)
    ax_main.plot(data)
    ax_main.set_xlim(0, len(data) - 1)

    for i, y in enumerate(rec_a):
        ax = fig.add_subplot(len(rec_a) + 1, 2, 3 + i * 2)
        ax.plot(y, 'r')
        ax.set_xlim(0, len(y) - 1)
        ax.set_ylabel("A%d" % (i + 1))
        plt.subplots_adjust(hspace=0.5)

    for i, y in enumerate(rec_d):
        ax = fig.add_subplot(len(rec_d) + 1, 2, 4 + i * 2)
        ax.plot(y, 'g')
        ax.set_xlim(0, len(y) - 1)
        ax.set_ylabel("D%d" % (i + 1))
        plt.subplots_adjust(hspace=0.5)

    plt.show()

    return rec_a, rec_d


def emd_plot(data, freq=100):
    end = len(data) / freq
    # t = np.arange(0, end, 0.01)
    s = np.array(data)
    emd = EMD()
    emd.emd(s)
    imfs, res = emd.get_imfs_and_residue()
    # imfs = emd(data)
    # rint(imfs)
    print(len(imfs))  # 7
    # print(type(imfs))  # <class 'numpy.ndarray'>
    max_freq_list = list()
    plt.figure(figsize=(10, 8))
    for i in range(len(imfs)):
        plt.subplot(len(imfs), 1, i+1)
        y_ppg, abs_y_ppg = signal_fft(imfs[i], freq)

        freq, max_ap = get_freq(y_ppg, abs_y_ppg)
        max_freq_list.append((freq, max_ap))
        plt.plot(y_ppg, abs_y_ppg)
        plt.ylabel("D%d" % (i + 1))
        plt.subplots_adjust(hspace=0.5)
    plt.show()

    # # 绘图 IMF
    # vis = Visualisation()
    # vis.plot_imfs(imfs=imfs, residue=res, t=t, include_residue=True)
    # # 绘制并显示所有提供的IMF的瞬时频率
    # vis.plot_instant_freq(t, imfs=imfs)
    # vis.show()

    return imfs, max_freq_list


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


def test_load_txt(path):
    data = pd.read_table(path, sep=',')
    data.columns = ['ir1', 'red1', 'ir2', 'red2', 'resp']
    return data


def split_data(ppg_data, resp_data, start, end):
    start_point = start
    end_point = end
    splited_ppg_data = ppg_data[start_point:end_point]
    splited_resp_data = resp_data[start_point:end_point]

    return splited_ppg_data, splited_resp_data


def signal_fft(data, freq):
    N = len(data)
    fs = freq
    df = fs / (N - 1)  # df分辨率
    # 构建频率数组
    f = [df * n for n in range(0, N)]
    Y = np.fft.fft(data) * 2 / N
    absY = [np.abs(x) for x in Y]

    # plt.plot(f, absY)
    # plt.show()
    return f, absY


def bandpass_filter(data, fs=500, start_fs=0.1, end_fs=0.8):
    winHz = [start_fs, end_fs]
    wn1 = 2 * winHz[0] / fs
    wn2 = 2 * winHz[1] / fs
    b, a = scisignal.butter(2, [wn1, wn2],
                            'bandpass')  # 截取频率[1，5]hz #这里假设采样频率为1000hz,信号本身最大的频率为500hz，要滤除10hz以下和400hz以上频率成分，即截至频率为10hz和400hz,则wn1=2*10/1000=0.02,wn2=2*400/1000=0.86。Wn=[0.02,0.86]#https://blog.csdn.net/weixin_37996604/article/details/82864680
    data = scisignal.filtfilt(b, a, data)  # data为要过滤的信号
    return data


def get_freq(freq_value, freq_energe):
    # 获取能量最大对应的频率
    N = int(len(freq_value) / 2)
    max_index = np.argmax(freq_energe[0:N])

    max_ap = np.max(freq_energe)
    freq = freq_value[max_index]

    # print("freq = {}, max_ap = {}".format(freq, max_ap))

    return freq, max_ap


def reverse(data):
    """
    反转波形
    """
    data_max = max(data)

    reversed_data = [data_max - _ for _ in data]

    return reversed_data


def plot_vmd(ppg_data, resp_data):
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
    freq, amplitude = signal_fft(fil_resp, 500)
    # get_freq(freq, amplitude)
    raw_disturb_freq, raw_disturb_amp = signal_fft(ppg_data, 500)

    plt.figure(figsize=(10, 8))
    plt.subplot(411)
    plt.plot(ppg_data)
    plt.plot(u[0], c='r')
    plt.title('raw_ppg')
    plt.subplot(412)
    plt.plot(results)
    plt.title('filted')
    plt.subplot(413)
    plt.plot(raw_disturb_freq, raw_disturb_amp, color='r')
    plt.title("respiratory")
    plt.subplot(414)
    plt.plot(freq, amplitude)
    plt.title("frequency")
    plt.subplots_adjust(hspace=0.8)
    plt.show()

    return resp

def cal_vmd(ppg_data):
    K = 8
    alpha = 5000
    tau = 1e-6
    vmd = VMD(K, alpha, tau)
    u, omega_K = vmd(ppg_data)

    vmd_filed = ppg_data - u[0]
    resp = u[0]
    # fil_data = bandpass_filter(results, start_fs=1, end_fs=5)
    fil_resp = bandpass_filter(resp, start_fs=0.1, end_fs=0.8)
    # freq, amplitude = signal_fft(fil_resp, 500)
    # get_freq(freq, amplitude)

    return resp, vmd_filed

def plot(**kwargs):
    plt.figure(figsize=(10, 8))
    count = 1
    for name, value in kwargs.items():
        plt.subplot(len(kwargs), 1, count)
        count += 1
        if len(value) == 2:
            plt.plot(value[0], value[1])
        else:
            plt.plot(value)
        plt.title(name)
        plt.subplots_adjust(hspace=0.8)
    plt.show()


def show_5disturb_3pulse():
    for path in glob.glob(r'../disturb/5disturb_3pulse/*.txt'):
        data = test_load_txt(path)
        ir2 = data['ir2']
        resp = data['resp']
        plot_vmd(ir2, resp)

def test_vmd(ppg_data):
    K = 8
    alpha = 5000
    tau = 1e-6
    vmd = VMD(K, alpha, tau)
    u, omega_K = vmd(ppg_data)

    results = ppg_data - u[0]
    resp = u[0]
    # fil_data = bandpass_filter(results, start_fs=1, end_fs=5)
    fil_resp = bandpass_filter(resp, start_fs=0.1, end_fs=0.8)
    # freq, amplitude = signal_fft(fil_resp, 500)
    # get_freq(freq, amplitude)

    return resp, fil_resp, results

def test1():
    dir = r'../disturb/0disturb_pulse/*.txt'
    for path in glob.glob(dir):
        # print(path)
        pulse_amp = float(os.path.split(path)[-1].split('_')[-1][:3])
        data = test_load_txt(path)
        ir2 = data['ir2']
        resp = data['resp']
        resp, buttered_resp, vmded_ppg = test_vmd(ir2)
        current = resp - buttered_resp
        f, absY = signal_fft(resp, 500)
        f1, absY1 = signal_fft(buttered_resp, 500)
        f2, absY2 = signal_fft(current, 500)
        f3, absY3 = signal_fft(vmded_ppg, 500)
        print('The mean of amplitude({}) is {}, the maximum is {}'.format(pulse_amp, round(np.mean(ir2), 2), round(np.max(ir2), 2)))
        plot(raw_data=ir2, vmded_ppg=vmded_ppg, resp=resp, buttered_resp=buttered_resp, current=current)
        plot(freq_raw=(f, absY), freq_buttered=(f1, absY1), freq_current=(f2, absY2), freq_vmd_ppg=(f3, absY3))

def test2():
    dir = r'../disturb/different_disturb/disturb1/*.txt'
    for path in glob.glob(dir):
        print(path)
        data = test_load_txt(path)
        ir2 = data['ir2']
        resp = data['resp']
        plot_vmd(ir2, resp)

def test_get_freq_amp():
    '''
    获取呼吸和ppg的频域幅值
    :return:
    '''
    root = r'../disturb/'
    dirs = os.listdir(root)
    writer = pd.ExcelWriter(r'E:/my_documents/实验记录/pulse_comparison_v2.xlsx', engine='openpyxl')
    for one_dir in dirs:
        if one_dir.endswith('disturb_pulse'):
            path = root + one_dir + '/*.txt'
            sheet_name = one_dir
            res = []
            for path in glob.glob(path):
                pulse_amp = float(os.path.split(path)[-1].split('_')[-1][:3])
                data = test_load_txt(path)
                ir2 = data['ir2']
                red2 = data['red2']
                resp = data['resp']
                resp, buttered_resp, vmded_ppg = test_vmd(ir2)
                # current = resp - buttered_resp
                # f, absY = signal_fft(resp, 500)
                resp_f, resp_absY = signal_fft(buttered_resp, 500)    # 呼吸频率
                max_resp_freq, max_resp_freq_amp = get_freq(resp_f, resp_absY)
                ppg_f, ppg_absY = signal_fft(vmded_ppg, 500)        # ppg频率
                max_ppg_freq, max_ppg_freq_amp = get_freq(ppg_f, ppg_absY)
                ratio = max_ppg_freq_amp / max_resp_freq_amp

                winsize = len(ir2)
                mean_ir = np.mean(ir2)
                mean_red = np.mean(red2)
                # 去除直流，直流定义为红光获红外平均值
                windata_ir_ac = [_ - mean_ir for _ in ir2]
                windata_red_ac = [_ - mean_red for _ in red2]
                # 求红光和红外的平方和，根据平方和求交流，直流交流处理方式
                ir_ac_pow = [math.pow(_, 2) for _ in windata_ir_ac]
                sum_ir_ac = np.sum(ir_ac_pow)
                ir_ac = math.sqrt(sum_ir_ac / winsize)

                red_ac_pow = [math.pow(_, 2) for _ in windata_red_ac]
                sum_red_ac = np.sum(red_ac_pow)
                red_ac = math.sqrt(sum_red_ac / winsize)
                # 求R值
                R = (red_ac * mean_ir) / (ir_ac * mean_red)

                # 求spo2 double result = 10.87 * newr * newr * newr - 52.97 * newr * newr + 25.24 * newr + 97.32;
                spo2_new = 10.87 * math.pow(R, 3) - 52.97 * math.pow(R, 2) + 25.24 * R + 97.32
                spo2 = -15.43 * math.pow(R, 2) - 15.15 * R + 112.6
                res.append([pulse_amp, round(max_resp_freq, 2), round(max_resp_freq_amp, 2), round(max_ppg_freq, 2), round(max_ppg_freq_amp, 2), round(R, 2), round(spo2, 2), 97, round(ratio, 2)])
                print("pulse{} resp_max_f={},resp_freq_max_amp={}; ppg_max_f={},ppg_freq_max_amp={};R={},spo2={},ratio={}".format(pulse_amp, round(max_resp_freq, 2), round(max_resp_freq_amp, 2), round(max_ppg_freq, 2), round(max_ppg_freq_amp, 2), round(R, 2), round(spo2, 2), round(ratio, 2)))
            df = pd.DataFrame(res)
            df.columns = ['pulse', 'resp_freq', 'resp_freq_amp', 'pulse_rate', 'pulse_rate_amp', 'R', 'spo2', 'real_spo2', 'ratio']
            # print(df)
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    writer.save()

def test_different_disturb():
    root = r'../disturb/different_disturb'
    dirs = os.listdir(root)
    print(dirs)
    writer = pd.ExcelWriter(r'E:/my_documents/实验记录/diferent_disturb_v2.xlsx', engine='openpyxl')
    for one_dir in dirs:
        pathes = root + '/' + one_dir + '/*.txt'
        sheet_name = one_dir
        print(pathes, sheet_name)
        res = []
        for path in glob.glob(pathes):
            print(path)
            file_name = os.path.split(path)[-1]
            if len(file_name) > 8:
                resp_rate = int(os.path.splitext(file_name)[0][-2:])
                print(resp_rate)
            else:
                resp_rate = int(os.path.split(path)[-1][:2])
            data = test_load_txt(path)
            ir2 = data['ir2']
            red2 = data['red2']
            resp = data['resp']
            resp, buttered_resp, vmded_ppg = test_vmd(ir2)
            resp_f, resp_absY = signal_fft(buttered_resp, 500)  # 呼吸频率
            max_resp_freq, max_resp_freq_amp = get_freq(resp_f, resp_absY)
            ppg_f, ppg_absY = signal_fft(vmded_ppg, 500)  # ppg频率
            max_ppg_freq, max_ppg_freq_amp = get_freq(ppg_f, ppg_absY)
            ratio = max_ppg_freq_amp / max_resp_freq_amp

            winsize = len(ir2)
            mean_ir = np.mean(ir2)
            mean_red = np.mean(red2)
            # 去除直流，直流定义为红光获红外平均值
            windata_ir_ac = [_ - mean_ir for _ in ir2]
            windata_red_ac = [_ - mean_red for _ in red2]
            # 求红光和红外的平方和，根据平方和求交流，直流交流处理方式
            ir_ac_pow = [math.pow(_, 2) for _ in windata_ir_ac]
            sum_ir_ac = np.sum(ir_ac_pow)
            ir_ac = math.sqrt(sum_ir_ac / winsize)

            red_ac_pow = [math.pow(_, 2) for _ in windata_red_ac]
            sum_red_ac = np.sum(red_ac_pow)
            red_ac = math.sqrt(sum_red_ac / winsize)
            # 求R值
            R = (red_ac * mean_ir) / (ir_ac * mean_red)

            # 求spo2 double result = 10.87 * newr * newr * newr - 52.97 * newr * newr + 25.24 * newr + 97.32;
            spo2_new = 10.87 * math.pow(R, 3) - 52.97 * math.pow(R, 2) + 25.24 * R + 97.32
            spo2 = -15.43 * math.pow(R, 2) - 15.15 * R + 112.6
            res.append([resp_rate, max_resp_freq, max_resp_freq_amp, max_ppg_freq, max_ppg_freq_amp, R, spo2, 97, ratio])
            print(
                "resp_rate{} resp_max_f={},resp_freq_max_amp={}; ppg_max_f={},ppg_freq_max_amp={};R={},spo2={},ratio={}".format(
                    resp_rate, round(max_resp_freq, 2), round(max_resp_freq_amp, 2), round(max_ppg_freq, 2),
                    round(max_ppg_freq_amp, 2), round(R, 2), round(spo2, 2), round(ratio, 2)))
        df = pd.DataFrame(res)
        df.columns = ['resp_rate', 'resp_freq', 'resp_freq_amp', 'pulse_rate', 'pulse_rate_amp', 'R', 'spo2',
                      'real_spo2', 'ratio']
        # print(df)
        df.to_excel(writer, sheet_name=sheet_name, index=False)
    writer.save()


def show(root_path):
    for path in glob.glob(root_path):
        print(path)
        data = test_load_txt(path)
        raw_ppg = data['ir2']
        filtered_ppg = bandpass_filter(raw_ppg, start_fs=0.1, end_fs=5)
        f, absY = signal_fft(filtered_ppg, freq=500)
        freq, amplitude = get_freq(f, absY)
        print('frequency = {}, amplitude = {}'.format(freq, amplitude))
        plot(raw_ppg=raw_ppg, filtered_ppg=filtered_ppg, frequency=(f, absY))

def show_different_pulse(root_path):
    for path in glob.glob(root_path):
        print(path)
        data = test_load_txt(path)
        raw_ppg = data['ir2']
        filtered_ppg = bandpass_filter(raw_ppg, start_fs=0.1, end_fs=5)
        f, absY = signal_fft(filtered_ppg, freq=500)
        # f, absY = signal_fft(raw_ppg, freq=500)
        peaks(f, absY)
        freq, amplitude = get_freq(f, absY)
        print('frequency = {}, amplitude = {}'.format(freq, amplitude))
        freq_peak_list = []
        # for i in range(1, len(absY)//2):
            # if absY[i-1] < absY[i] and absY[i] > absY[i+1]:
            #     index = absY.index(absY[i])
            #     freq_peak_list.append(f[index])

        # print('all peaks is ', freq_peak_list)
        plot(raw_ppg=raw_ppg, filtered_ppg=filtered_ppg, frequency=(f, absY))

def cal_spo2(root_path):
    for path in glob.glob(root_path):
        print(path)
        data = test_load_txt(path)
        ir2 = data['ir2']
        red2 = data['red2']
        ir2_resp, ir2_filtered = cal_vmd(ir2)
        red2_resp, red2_filtered = cal_vmd(red2)
        ir2_resp_wind = ir2_resp[0:2000]
        red2_resp_wind = red2_resp[0:2000]
        windata_ir = ir2_filtered[0:2000]
        windata_red = red2_filtered[0:2000]
        winsize = len(windata_ir)
        mean_ir = np.mean(ir2_resp_wind)
        mean_red = np.mean(red2_resp_wind)
        # 去除直流，直流定义为红光获红外平均值
        # windata_ir_ac = [_ - mean_ir for _ in windata_ir]
        # windata_red_ac = [_ - mean_red for _ in windata_red]

        # 求红光和红外的平方和，根据平方和求交流，直流交流处理方式
        # ir_ac_pow = [math.pow(_, 2) for _ in windata_ir_ac]
        ir_ac_pow = [math.pow(_, 2) for _ in windata_ir]

        sum_ir_ac = np.sum(ir_ac_pow)
        ir_ac = math.sqrt(sum_ir_ac / winsize)

        # red_ac_pow = [math.pow(_, 2) for _ in windata_red_ac]
        red_ac_pow = [math.pow(_, 2) for _ in windata_red]

        sum_red_ac = np.sum(red_ac_pow)
        red_ac = math.sqrt(sum_red_ac / winsize)
        # 求R值
        R = (red_ac * mean_ir) / (ir_ac * mean_red)

        # spo2 = 10.87*math.pow(R, 3)-52.97*math.pow(R, 2)+25.24*R+97.32
        # spo2 = -15.43 * math.pow(R, 2) - 15.15 * R + 111.6
        new_spo2 = -15.43 * math.pow(R, 2) - 15.15 * R + 111.6

        # spo2 = 10.87 * math.pow(R, 3) - 52.97 * math.pow(R, 2) + 25.24 * R + 97.32
        print('spo2--------------', new_spo2)


def r_test():
    root_path = '../disturb/0disturb_pulse/*.txt'
    for path in glob.glob(root_path):
        pulse_amp = float(os.path.split(path)[-1].split('_')[-1][:3])
        data = test_load_txt(path)
        windata_ir = data['ir2']
        windata_red = data['red2']
        winsize = len(windata_ir)

        mean_ir = np.mean(windata_ir)
        mean_red = np.mean(windata_red)
        # 去除直流，直流定义为红光获红外平均值
        windata_ir_ac = [_ - mean_ir for _ in windata_ir]
        windata_red_ac = [_ - mean_red for _ in windata_red]
        # 求红光和红外的平方和，根据平方和求交流，直流交流处理方式
        ir_ac_pow = [math.pow(_, 2) for _ in windata_ir_ac]
        sum_ir_ac = np.sum(ir_ac_pow)
        ir_ac = math.sqrt(sum_ir_ac / winsize)

        red_ac_pow = [math.pow(_, 2) for _ in windata_red_ac]
        sum_red_ac = np.sum(red_ac_pow)
        red_ac = math.sqrt(sum_red_ac / winsize)
        # 求R值
        R = (red_ac * mean_ir) / (ir_ac * mean_red)

        # 求spo2 double result = 10.87 * newr * newr * newr - 52.97 * newr * newr + 25.24 * newr + 97.32;
        # spo2 = 10.87*math.pow(R, 3)-52.97*math.pow(R, 2)+25.24*R+97.32
        spo2 = -15.43 * math.pow(R, 2) - 15.15 * R + 112.6

        print("The pulse amplitude is {} : spo2 = {}, R = {}".format(pulse_amp, round(spo2, 2), round(R, 2)))


def peaks(freq, absY):
    absY = np.array(absY)[:len(absY)//2]
    peaks_array = signal.find_peaks(absY, height=(absY.max() * 0.01))[0]
    peaks = []
    for i in peaks_array:
        peaks.append((round(freq[i], 4), round(absY[i], 2)))
    print(peaks)

def my_logic(data):
    disturb = None         # 呼吸干扰
    pulse = None           # 脉冲幅值
    ir2 = data['ir2']
    red2 = data['red2']
    resp = data['resp']
    winsize = len(ir2)
    resp, buttered_resp, vmded_ppg = test_vmd(ir2)
    resp_f, resp_absY = signal_fft(buttered_resp, 500)  # 呼吸频率
    max_resp_freq, max_resp_freq_amp = get_freq(resp_f, resp_absY)
    ppg_f, ppg_absY = signal_fft(vmded_ppg, 500)  # ppg频率
    max_ppg_freq, max_ppg_freq_amp = get_freq(ppg_f, ppg_absY)
    print('max_resp_freq={},max_resp_freq_amp={}'.format(max_resp_freq, max_resp_freq_amp))
    if max_resp_freq > 2:
        max_resp_freq = 1.0

    mean_ir = np.mean(ir2)
    mean_red = np.mean(red2)
    # 去除直流，直流定义为红光获红外平均值
    windata_ir_ac = [_ - mean_ir for _ in ir2]
    windata_red_ac = [_ - mean_red for _ in red2]
    # 求红光和红外的平方和，根据平方和求交流，直流交流处理方式
    ir_ac_pow = [math.pow(_, 2) for _ in windata_ir_ac]
    sum_ir_ac = np.sum(ir_ac_pow)
    ir_ac = math.sqrt(sum_ir_ac / winsize)

    red_ac_pow = [math.pow(_, 2) for _ in windata_red_ac]
    sum_red_ac = np.sum(red_ac_pow)
    red_ac = math.sqrt(sum_red_ac / winsize)
    # 求R值
    R = (red_ac * mean_ir) / (ir_ac * mean_red)

    # 求spo2 double result = 10.87 * newr * newr * newr - 52.97 * newr * newr + 25.24 * newr + 97.32;
    # spo2_v1 = 10.87*math.pow(R, 3)-52.97*math.pow(R, 2)+25.24*R+97.32
    # spo2 = -15.43 * math.pow(R, 2) - 15.15 * R + 112.6
    if max_resp_freq_amp < 500:
        if max_ppg_freq_amp < 100:
            pulse = 0
        else:
            pulse = 0.000635455 * max_ppg_freq_amp - 0.01391
        disturb = 0
        resp_rate = 0                         # 呼吸干扰为0，呼吸为0
        spo2 = -15.43 * math.pow(R, 2) - 15.15 * R + 111.6
        # spo2 = -15.43 * math.pow(R, 2) - 15.15 * R + 111.6 - 1.03102 * pulse + 0.9571
        pulse_rate = int(max_ppg_freq * 60)

    elif 500 <= max_resp_freq_amp < 2000:
        disturb = 1
        if max_ppg_freq == max_resp_freq:
            pulse = disturb / 10
        elif max_ppg_freq < 0.5 or max_ppg_freq > 5:
            pulse = 0.1
        else:
            pulse = 0.00089441 * max_ppg_freq_amp + 0.09726
        resp_rate = int(max_resp_freq * 60)
        pulse_rate = int(max_ppg_freq * 60)
        spo2 = -15.43 * math.pow(R, 2) - 15.15 * R + 111.6 - 4.18612 * pulse + 12.0813

    elif 2000 <= max_resp_freq_amp < 3200:
        disturb = 2
        if max_ppg_freq == max_resp_freq:
            pulse = disturb / 10
        elif max_ppg_freq < 0.5 or max_ppg_freq > 5:
            pulse = 0.1
        else:
            pulse = 0.000891282 * max_ppg_freq_amp + 0.04567
        resp_rate = int(max_resp_freq * 60)
        pulse_rate = int(max_ppg_freq * 60)
        spo2 = -15.43 * math.pow(R, 2) - 15.15 * R + 111.6 - 3.652 * pulse + 15.98152

    elif 3200 <= max_resp_freq_amp < 4200:
        disturb = 3
        if max_ppg_freq == max_resp_freq:
            pulse = disturb / 10
        elif max_ppg_freq < 0.5 or max_ppg_freq > 5:
            pulse = 0.1
        else:
            pulse = 0.00109 * max_ppg_freq_amp - 0.00781
        resp_rate = int(max_resp_freq * 60)
        pulse_rate = int(max_ppg_freq * 60)
        spo2 = -15.43 * math.pow(R, 2) - 15.15 * R + 111.6 - 2.38348 * pulse + 15.99768

    elif 4200 <= max_resp_freq_amp < 9000:
        disturb = 4
        if max_ppg_freq == max_resp_freq:
            pulse = disturb / 10
        elif max_ppg_freq < 0.5 or max_ppg_freq > 5:
            pulse = 0.1
        else:
            pulse = 0.00061119 * max_ppg_freq_amp + 0.02497
        resp_rate = int(max_resp_freq * 60)
        pulse_rate = int(max_ppg_freq * 60)
        spo2 = -15.43 * math.pow(R, 2) - 15.15 * R + 111.6 - 1.25704 * pulse + 15.87017

    else:
        disturb = 5
        if max_ppg_freq == max_resp_freq:
            pulse = disturb / 10
        elif max_ppg_freq < 0.5 or max_ppg_freq > 5:
            pulse = 0.1
        else:
            pulse = 0.000623342 * max_ppg_freq_amp + 0.00749
        resp_rate = int(max_resp_freq * 60)
        pulse_rate = int(max_ppg_freq * 60)
        spo2 = -15.43 * math.pow(R, 2) - 15.15 * R + 112.6 - 1.76694 * pulse + 16.12595
    if spo2 > 100:
        spo2 = 100
    if pulse_rate >= 120:
        pulse_rate = 60

    pulse = round(pulse, 1)
    spo2 = round(spo2)

    print("disturb={},pulse={},resp_rate={},pulse_rate={},spo2={}".format(disturb, pulse, resp_rate, pulse_rate, spo2))

def my_logic_test():
    root = r'../disturb/'
    dirs = os.listdir(root)
    for one_dir in dirs:
        if one_dir.endswith('disturb_pulse'):
            pathes = root + one_dir + '/*.txt'
            print(pathes)
            for path in glob.glob(pathes):
                pulse_amp = float(os.path.split(path)[-1].split('_')[-1][:3])
                print('real_pulse=', pulse_amp)
                data = test_load_txt(path)
                my_logic(data)

def my_logic_test_other():
    pathes = r'../disturb/different_disturb/disturb5/*.txt'
    for path in glob.glob(pathes):
        # pulse_amp = float(os.path.split(path)[-1].split('_')[-1][:3])
        # print('real_pulse=', pulse_amp)
        print(path)
        data = test_load_txt(path)
        my_logic(data)

def temp():
    path = r'../disturb/0disturb_pulse/2022_06_21_09_36_39_1.86.txt'
    pulse_amp = float(os.path.split(path)[-1].split('_')[-1][:3])
    data = test_load_txt(path)
    ir2 = data['ir2']
    red2 = data['red2']
    resp = data['resp']
    resp, buttered_resp, vmded_ppg = test_vmd(ir2)
    # current = resp - buttered_resp
    # f, absY = signal_fft(resp, 500)
    resp_f, resp_absY = signal_fft(buttered_resp, 500)  # 呼吸频率
    max_resp_freq, max_resp_freq_amp = get_freq(resp_f, resp_absY)
    ppg_f, ppg_absY = signal_fft(vmded_ppg, 500)  # ppg频率
    max_ppg_freq, max_ppg_freq_amp = get_freq(ppg_f, ppg_absY)

    winsize = len(ir2)
    mean_ir = np.mean(ir2)
    mean_red = np.mean(red2)
    # 去除直流，直流定义为红光获红外平均值
    windata_ir_ac = [_ - mean_ir for _ in ir2]
    windata_red_ac = [_ - mean_red for _ in red2]
    # 求红光和红外的平方和，根据平方和求交流，直流交流处理方式
    ir_ac_pow = [math.pow(_, 2) for _ in windata_ir_ac]
    sum_ir_ac = np.sum(ir_ac_pow)
    ir_ac = math.sqrt(sum_ir_ac / winsize)

    red_ac_pow = [math.pow(_, 2) for _ in windata_red_ac]
    sum_red_ac = np.sum(red_ac_pow)
    red_ac = math.sqrt(sum_red_ac / winsize)
    # 求R值
    R = (red_ac * mean_ir) / (ir_ac * mean_red)

    # 求spo2 double result = 10.87 * newr * newr * newr - 52.97 * newr * newr + 25.24 * newr + 97.32;
    spo2_new = 10.87 * math.pow(R, 3) - 52.97 * math.pow(R, 2) + 25.24 * R + 97.32
    spo2 = -15.43 * math.pow(R, 2) - 15.15 * R + 112.6
    # res.append([pulse_amp, max_resp_freq, max_resp_freq_amp, max_ppg_freq, max_ppg_freq_amp, R, spo2, 97])
    # print(
    #     "pulse{} resp_max_f={},resp_freq_max_amp={}; ppg_max_f={},ppg_freq_max_amp={};R={},spo2={},spo2_new={}".format(
    #         pulse_amp, round(max_resp_freq, 2), round(max_resp_freq_amp, 2), round(max_ppg_freq, 2),
    #         round(max_ppg_freq_amp, 2), round(R, 2), round(spo2, 2), round(spo2_new, 2)))

    print(
        "pulse{} resp_max_f={},resp_freq_max_amp={}; ppg_max_f={},ppg_freq_max_amp={};R={},spo2={},spo2_new={}".format(
            pulse_amp, max_resp_freq, max_resp_freq_amp, max_ppg_freq,
            max_ppg_freq_amp, R, spo2, spo2_new))



if __name__ == '__main__':

    # root_path = r'../disturb/5disturb_0pulse/*.txt'      # 脉冲幅值为0，呼吸率不同
    # root_path = r'../disturb/different_disturb/disturb5/*.txt'   # 不同干扰
    # root_path = r'../disturb/5disturb_3pulse/*.txt'      # 呼吸率不同
    root_path = r'../disturb/1disturb_pulse/*.txt'       # 脉冲幅值不同
    # root_path = r'../disturb/different_spo2/1disturb_3pulse/*.txt'
    # show_5disturb_3pulse()
    # show(root_path)
    # show_different_pulse(root_path)
    # cal_spo2(root_path)
    # test1()
    # test2()
    # r_test()
    # test_get_freq_amp()
    # temp()
    # my_logic_test()
    # my_logic_test_other()        # 测试
    test_different_disturb()