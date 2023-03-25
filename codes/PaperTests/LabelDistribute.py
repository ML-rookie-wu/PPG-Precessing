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

sns.set(style="darkgrid")

path = r"D:\my_projects_V1\my_projects\PPG_V1\data\BR\LSTMDataset\features_10_new.csv"
df = pd.read_csv(path, sep=",")
lable = df["label"]
print(lable)
lable.value_counts()
sns.countplot(x="label", data=df).set_title("Distribution of outcome")
# plt.title("Distribution of outcome")
plt.savefig(os.path.join(PAPER_FIGURE_PATH, "结果分布"), dpi=300, bbox_inches="tight")
plt.show()