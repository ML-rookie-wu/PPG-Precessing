# -*- coding: utf-8 -*-

"""
@author: Mark Wu
@file: br_test1.py
@time: 2023/3/1 9:37
"""

import pandas as pd
import matplotlib.pyplot as plt
from codes.utils.MyFilters import bandpass_filter


path = r"D:\my_projects_V1\my_projects\采集软件\record.txt"
# path = r"D:\my_projects_V1\my_projects\PPG_V1\data\BR\apnea\wu\20230301205746.txt"
data = pd.read_table(path, sep="\t")
# data = data[5000:25000]
print(data.columns)
ir = data[" IIr_Sample"]
resp = data[" Resp_Sample"]
pr = data[" PR"]
spo2 = data[" SPO2"]
spo2_real = data[" SPO2_Real"]
rrInterval = data[" rrInterval"]
# signal = data[" Signal"]
resp_butter = bandpass_filter(resp, fs=100, start_fs=0.1, end_fs=0.8)

plt.subplot(211)
plt.plot(ir)
plt.subplot(212)
plt.plot(resp)
plt.show()

plt.subplot(411)
plt.plot(pr)
plt.subplot(412)
plt.plot(spo2)
plt.subplot(413)
plt.plot(resp)
plt.subplot(414)
plt.plot(ir)
plt.show()
