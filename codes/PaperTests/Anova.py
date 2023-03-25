# -*- coding: utf-8 -*-

"""
@author: Mark Wu
@file: Anova.py
@time: 2023/3/22 21:02
"""

import os

#可视化包
import seaborn as sns
# from IPython.display import Image
# import pydotplus
import matplotlib.pyplot as plt
sns.set()
# os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
# plt.rcParams['font.sans-serif'] = ['Time New Roman']
# plt.rcParams['font.serif'] = ['simhei']
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
plt.rc("font", family="Time New Roman")
sns.set_style("darkgrid", {"font.sans-serif": ['simhei', 'Droid Sans Fallback']})

#数学包
from scipy import stats
import pandas as pd
import numpy as np

#统计检验
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import MultiComparison
from codes.PaperTests import PAPER_FIGURE_PATH


#读取
path = r"D:\my_projects_V1\my_projects\PPG_V1\data\BR\LSTMDataset\features_10_new.csv"
df_0 = pd.read_csv(path, sep=",")
column_name = df_0.columns
# df_0 = pd.DataFrame(iris.data, columns=['SpealLength','Spealwidth','PetalLength','Petalwidth'])
# df_0['target'] = df_0["label"]


# 封装 双变量-单因素方差分析
def my_oneWayAnova(df, cata_name, num_name, alpha_anova=0.05, alpha_tukey=0.05):
    df[cata_name] = df[cata_name].astype('str')

    s1 = df[cata_name]
    s2 = df[num_name]

    fml = num_name + '~C(' + cata_name + ')'

    model = ols(fml, data=df).fit()
    anova_table_1 = anova_lm(model, typ=2).reset_index()
    p1 = anova_table_1.loc[0, 'PR(>F)']

    # 输出 ： 是否相等【不等式序列】
    if p1 > alpha_anova:
        print('组间【无】显著差异')
    else:
        print('组间【有】显著差异')
        # 输出不等式

    # 输出： 统计结果表（均值，分位数，差异组）
    df_p1 = df.groupby([cata_name])[num_name].describe()

    # 输出： Tudey 多重比较
    mc = MultiComparison(df[num_name], df[cata_name])
    df_smry = mc.tukeyhsd(alpha=alpha_tukey).summary()
    m = np.array(df_smry.data)
    df_p2 = pd.DataFrame(m[1:], columns=m[0])

    # 输出 ：分类直接的大小差异显著性
    df_p1_sub = df_p1[['mean']].copy()
    df_p1_sub.sort_values(by='mean', inplace=True)

    output_list = []

    for x in range(1, len(df_p1_sub.index)):
        if (df_p2.loc[((df_p2.group1 == df_p1_sub.index[x - 1]) & (df_p2.group2 == df_p1_sub.index[x])) |
                      ((df_p2.group1 == df_p1_sub.index[x]) & (df_p2.group2 == df_p1_sub.index[x - 1])),
                      'reject'].iloc[0]) == "True":
            smb = '<'
        else:
            smb = '<='
        if x == 1:
            output_list.append(df_p1_sub.index[x - 1])
            output_list.append(smb)
            output_list.append(df_p1_sub.index[x])
        else:
            output_list.append(smb)
            output_list.append(df_p1_sub.index[x])
    out_sentence = ' '.join(output_list)
    print(out_sentence)

    # 输出： 箱线图
    # 分布可视化boxplot
    # plt.figure(figsize=(10, 8))
    # sns.boxplot(x=cata_name, y=num_name, data=df)  # ,order=df_p1_sub.index
    # sns.set_theme(font="Time New Roman")
    # plt.title("raw_mean", y=1.01)
    # # plt.savefig(os.path.join(PAPER_FIGURE_PATH, "raw_mean"), dpi=300, bbox_inches="tight")
    # plt.show()

    return df_p1, df_p2

def main(df_0):
    fig = plt.figure(figsize=(12, 16))
    for i in range(1, len(column_name)):
        print("----------------%s-------------" % column_name[i])
        df1, df2 = my_oneWayAnova(df_0, cata_name="label", num_name=column_name[i])
        print(df1)
        print(df2)
        # plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(4, 3, i)
        sns.boxplot(x="label", y=column_name[i], data=df_0, ax=ax, meanline=True, showmeans=True)  # ,order=df_p1_sub.index
        # ax.set_title(column_name[i])
        sns.set_theme(font="Time New Roman")
        plt.subplots_adjust(wspace=0.5, hspace=0.3)
    plt.savefig(os.path.join(PAPER_FIGURE_PATH, "特征分析"), dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == '__main__':


    # df1, df2 = my_oneWayAnova(df_0, cata_name="label", num_name="raw_mean")
    # print(df1)
    # print(df2)

    main(df_0)