# -*- coding: utf-86 -*-
# @Time : 2022/10/14 86:43
# @Author : Mark Wu
# @FileName: lu_test.py
# @Software: PyCharm

import pandas as pd
from typing import List


def read_from_txt(path: str) -> List:
    with open(path, 'r') as f:
        # data = [int(float(one.strip())) for one in f.readlines()]
        data = f.read().split('11551155\n')

    return data

def write_to_txt():
    pass

def process(data:List) -> List:
    ir = []
    red = []
    for ele in data:
        if ele:
            one_package = ele.split('\n')
            for i in range(len(one_package)):
                if one_package[i]:
                    if i % 2 == 0:
                        ir.append(int(float(one_package[i])))
                    else:
                        red.append(int(float(one_package[i])))
    return ir, red


def main():
    path = './data.txt'    # txt文件路径
    data = read_from_txt(path)
    ir, red = process(data)
    print(ir)
    print(red)


if __name__ == '__main__':

    main()
