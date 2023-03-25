# -*- coding: utf-8 -*-

"""
@author: Mark Wu
@file: read_dat.py
@time: 2023/2/26 10:03
"""

import pandas as pd
import matplotlib.pyplot as plt
from codes.utils.MyFilters import bandpass_filter, vmd
from codes.utils.GetFFT import signal_fft, get_freq


data = pd.read_csv(r"D:\DownloadPath\bidmc_02_Signals.csv")
print(type(data))
print(data.columns)
ir = data[" PLETH"][0:7500]
resp = data[" RESP"][0:7500]

butter_ir = bandpass_filter(ir, fs=125, start_fs=0.1, end_fs=5)
butter_resp = bandpass_filter(resp, fs=125, start_fs=0.1, end_fs=1)

result, u, resp_vmd = vmd(ir)
buttered_vmd = bandpass_filter(resp_vmd, fs=125, start_fs=0.1, end_fs=0.7)
plt.plot(buttered_vmd)
plt.show()
f_vmd, abs_vmd = signal_fft(buttered_vmd, freq=125)
f, amp = get_freq(f_vmd, abs_vmd)
print("f=", f)

f_resp, abs_resp = signal_fft(butter_resp, freq=125)

freq, max_amp = get_freq(f_resp, abs_resp)
print("freq=", freq)
plt.plot(f_resp, abs_resp)
plt.show()

plt.figure(figsize=(10, 8))
plt.subplot(411)
plt.plot(ir)
plt.subplot(412)
plt.plot(butter_ir)
plt.subplot(413)
plt.plot(resp)
plt.subplot(414)
plt.plot(butter_resp)
plt.show()

# content = f.read()
