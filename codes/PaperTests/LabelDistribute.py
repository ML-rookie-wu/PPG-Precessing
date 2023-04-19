# -*- coding: utf-8 -*-

"""
@author: Mark Wu
@file: LabelDistribute.py
@time: 2023/3/23 9:55
"""

import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from codes.PaperTests import PAPER_FIGURE_PATH

plt.rcParams['font.sans-serif'] = ['SongNTR']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# sns.set(style="darkgrid")
# plt.style.use('seaborn-darkgrid')

# print(plt.style.available)
path = r"D:\my_projects_V1\my_projects\PPG_V1\data\BR\LSTMDataset\features_10_new.csv"
# path = r"D:\my_projects_V1\my_projects\PPG_V1\data\respiration2\features.csv"
df = pd.read_csv(path, sep=",")
lable = df["label"]
# print(lable)
lable.value_counts()
sns.countplot(x="label", data=df).set_title("Distribution of outcome", fontsize=10.5)
# plt.title("Distribution of outcome")
plt.ylabel("Count", fontsize=10.5)
plt.xlabel("Label", fontsize=10.5)
plt.xticks(fontsize=10.5)
plt.yticks(fontsize=10.5)


plt.savefig(os.path.join(PAPER_FIGURE_PATH, "结果分布1"), dpi=300, bbox_inches="tight")
plt.show()