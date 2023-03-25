# -*- coding: utf-8 -*-

"""
@author: Mark Wu
@file: LMSTest1.py
@time: 2023/1/12 11:25
"""

import numpy as np
import matplotlib.pyplot as plt



# 信号加噪
def awgn(x, snr):
    snr = 10 ** (snr / 10.0)
    xpower = np.sum(np.abs(x) ** 2) / len(x)
    npower = xpower / snr
    if type(x[0]) != np.complex128:
        return x + np.random.randn(len(x)) * np.sqrt(npower)
    else:
        return x + np.random.randn(len(x)) * np.sqrt(npower / 2) + 1j * np.random.randn(len(x)) * np.sqrt(npower / 2)


def lmsFunc(xn, dn, M, mu):
    itr = len(xn)
    en = np.zeros((itr, 1))
    W = np.zeros((M, itr))
    for k in range(M, itr):
        if k==20:
            x = xn[k-1::-1]
        else:
            x = xn[k-1:k-M-1:-1]
        try:
            y = np.dot(W[:, k - 2], x)
            print(y)
        except:
            pass
        en[k-1] = dn[k-1] - y
        W[:, k-1] = W[:, k - 2] + 2 * mu * en[k-1] * x

    yn = np.ones(xn.shape) * np.nan
    for k in range(M, len(xn) ):
        if k == 20:
            x = xn[k - 1::-1]
        else:
            x = xn[k - 1:k - M - 1:-1]
        yn[k] = np.dot(W[:, -2], x)

    return yn, W, en


if __name__ == '__main__':
    fs = 1
    f0 = 0.02
    n = 1000
    t = np.arange(n)/fs
    xs = np.cos(2*np.pi*f0*t)
    ws = awgn(xs, 20)
    # data1 = scio.loadmat('xs.mat')
    # data2 = scio.loadmat('ws.mat')
    # xs = data1['xs'].flatten()
    # ws = data2['ws'].flatten()

    M = 20
    xn = ws
    dn = xs
    mu = 0.001
    yn, W, en = lmsFunc(xn, dn, M, mu)

    plt.figure()
    plt.subplot(211)
    plt.plot(t, ws)
    plt.subplot(212)
    plt.plot(t, yn)

    plt.figure()
    plt.plot(en)
    plt.show()
