# -*- coding: utf-86 -*-
# @Time : 2022/11/1 11:04
# @Author : Mark Wu
# @FileName: spo2_test.py
# @Software: PyCharm

import json
import os
from codes.read_data import Process, cal_spo2


def test(data_dir=None, save_path=None):
    p = Process()
    res = {}
    if data_dir is None:
        data_dir = p.data_dir

    for root_path, dir_names, filenames in os.walk(data_dir):
        # print(dir_names)
        for file_name in filenames:
            if file_name.find("txt") < 0:
                continue

            print(os.path.join(root_path, file_name))

            real_spo2 = int(file_name.split('_')[-1].split(".")[0])
            if real_spo2 <= 30:
                real_spo2 = 97

            res[real_spo2] = []
            regr = []
            bivariate_regr = []
            ternary_regr = []
            file_path = os.path.join(root_path, file_name)
            ir2, red2 = p.read_from_file(file_path)

            start = 5000
            step = 4000
            end = start + step
            while end <= len(ir2):
                vmded_ir2, ir2_baseline = p.preprocess(ir2[start: end])
                vmded_red2, red2_baseline = p.preprocess(red2[start: end])
                R = p.get_R(vmded_ir2, vmded_red2, ir2_baseline, red2_baseline)
                spo2_third, spo2_second, spo2 = cal_spo2(R)
                regr.append(spo2)
                bivariate_regr.append(spo2_second)
                ternary_regr.append(spo2_third)
                start += 1000
                end = start + step
            res[real_spo2].append(regr)
            res[real_spo2].append(bivariate_regr)
            res[real_spo2].append(ternary_regr)
    print(res)
    if save_path is not None:
        with open(save_path, 'w') as f:
            json.dump(res, f)


# test(r'D:\works\PPG\data\spo2_compare\2disturb_5pulse', save_path=r'D:\works\PPG\results\2disturb_5pulse.json')
# all_dir = list(os.walk(r'D:\works\PPG\data\spo2_compare\2disturb_5pulse'))
# print(len(all_dir))

# for root_path, dir_path, filenames in os.walk(r'D:\works\PPG\data\spo2_compare'):
#     for filename in filenames:
#
#         print(os.path.join(root_path, filename))
