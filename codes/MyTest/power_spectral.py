#coding:utf-8

from scipy.fftpack import fft, fftshift, ifft
from scipy.fftpack import fftfreq
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


def test():
    fs = 1000
    #采样点数
    num_fft = 1024

    """
    生成原始信号序列
    
    在原始信号中加上噪声
    np.random.randn(t.size)
    """
    t = np.arange(0, 1, 1/fs)
    f0 = 100
    f1 = 200
    x = np.cos(2*np.pi*f0*t) + 3*np.cos(2*np.pi*f1*t) + np.random.randn(t.size)
    print('len----------', len(x))
    N = len(x)

    plt.figure(figsize=(15, 12))
    ax=plt.subplot(511)
    ax.set_title('original signal')
    plt.tight_layout()
    plt.plot(x)

    """
    FFT(Fast Fourier Transformation)快速傅里叶变换
    """
    Y = fft(x)
    Y = np.abs(Y)

    ax=plt.subplot(512)
    ax.set_title('fft transform')
    plt.plot(20*np.log10(Y[:N]))

    """
    功率谱 power spectrum
    直接平方
    """
    ps = Y**2 / N
    ax=plt.subplot(513)
    ax.set_title('direct method')
    plt.plot(20*np.log10(ps[:N//2]))

    """
    相关功谱率 power spectrum using correlate
    间接法
    """
    cor_x = np.correlate(x, x, 'same')
    cor_X = fft(cor_x, N)
    ps_cor = np.abs(cor_X)
    ps_cor = ps_cor / np.max(ps_cor)
    ax=plt.subplot(514)
    ax.set_title('indirect method')
    plt.plot(20*np.log10(ps_cor[:N//2]))

    df = fs / (N - 1)  # df分辨率
    # 构建频率数组
    f = [df * n for n in range(0, N)]
    Y = np.fft.fft(x) * 2 / 1000
    absY = [np.abs(x) for x in Y]
    ax = plt.subplot(515)
    ax.set_title("fft")
    plt.plot(f, absY)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    test()