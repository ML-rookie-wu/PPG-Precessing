#coding:utf-8
import matplotlib.pyplot as plt
import numpy as np


def signal_fft(data, freq=500):
    N = len(data)
    fs = freq
    df = fs / (N - 1)
    f = [df * n for n in range(0, N)]
    Y = np.fft.fft(data) * 2 / N
    absY = [np.abs(x) for x in Y]
    return f, absY


def vmd_plot(ir2, imfs):
    fig = plt.figure(figsize=(16, 12))
    ax_main = fig.add_subplot(len(imfs) + 2, 1, 1)
    ax_main.plot(ir2)
    ax_main.set_xlim(0, len(ir2) - 1)

    for i, x in enumerate(imfs):
        ax = fig.add_subplot(len(imfs) + 2, 2, 5 + i * 2)
        ax.plot(x)
        ax.set_xlim(0, len(x) - 1)
        ax.set_ylabel("imf%d" % (i + 1))
        plt.subplots_adjust(hspace=0.8)

    for i, y in enumerate(imfs):
        ax = fig.add_subplot(len(imfs) + 2, 2, 6 + i * 2)
        f, absY = signal_fft(y)
        ax.plot(f, absY)
        ax.set_ylabel("freq%d" % (i + 1))
        plt.subplots_adjust(hspace=0.8)
    plt.show()


if __name__ == '__main__':

    ir2 = None
    imfs = None
    vmd_plot(ir2, imfs)