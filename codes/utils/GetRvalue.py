#coding:utf-8

import math
import numpy as np


def get_R(filtered_ir, filtered_red, ir, red):
    """
    :param filtered_ir: 滤波后的ir信号
    :param filtered_red: 滤波后的red信号
    :param ir: 原始的ir信号或者ir信号的基线
    :param red: 原始的red信号或者red信号的基线
    :return: R值
    """
    winsize = len(ir)
    # mean_ir = np.mean(ir)
    # mean_red = np.mean(red)
    # # 去除直流，直流定义为红光获红外平均值
    # windata_ir_ac = [_ - mean_ir for _ in ir]
    # windata_red_ac = [_ - mean_red for _ in red]
    # # 求红光和红外的平方和，根据平方和求交流，直流交流处理方式
    # ir_ac_pow = [math.pow(_, 2) for _ in windata_ir_ac]
    # sum_ir_ac = np.sum(ir_ac_pow)
    # ir_ac = math.sqrt(sum_ir_ac / winsize)
    #
    # red_ac_pow = [math.pow(_, 2) for _ in windata_red_ac]
    # sum_red_ac = np.sum(red_ac_pow)
    # red_ac = math.sqrt(sum_red_ac / winsize)

    # 直流
    ir_dc = np.mean(ir)
    red_dc = np.mean(red)

    # 交流
    ir_ac = math.sqrt(sum(x ** 2 for x in filtered_ir)) / winsize
    red_ac = math.sqrt(sum(y ** 2 for y in filtered_red)) / winsize

    # 求R值
    # R = (red_ac * mean_ir) / (ir_ac * mean_red)

    R = (red_ac * ir_dc) / (ir_ac * red_dc)

    return R