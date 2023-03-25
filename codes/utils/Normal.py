# -*- coding: utf-8 -*-

"""
@author: Mark Wu
@file: Normal.py
@time: 2023/1/11 16:19
"""

import numpy as np


def normalization(data):
    """标准化"""
    mean_value = np.mean(data)
    std_value = np.std(data)
    normal = [(x - mean_value) / std_value for x in data]
    return normal

