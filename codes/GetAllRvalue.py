#coding:utf-8

from codes.utils.MyFilters import MyProcess
from codes.utils.GetFileData import travel_dir, read_from_file
from codes.utils.GetRvalue import get_R
import os
import pandas as pd


def save_to_excel(save_path, data):
    # file_name = "R.xlsx"
    # path = os.path.join(self.results_path, file_name)
    if save_path is not None:
        writer = pd.ExcelWriter(save_path, engine="openpyxl")
        df = pd.DataFrame(data)
        df.columns = ['R', 'spo2_value']
        df.to_excel(writer, index=False)
        writer.save()
    else:
        raise Exception("Need a save path,but there is not")


def get_R_list(parent_dir):
    all_files = travel_dir(parent_dir)
    filter_func = MyProcess('dwt')
    res = []
    for file_path in all_files:
        spo2_value = int(os.path.split(file_path)[1].split(".")[0].split("_")[1])
        ir2, red2 = read_from_file(file_path)
        start = 1000
        step = 500
        window = 4000
        end = start + step
        while end <= len(ir2):
            # vmd
            # vmded_ir2, ir2_disturb = self.preprocess(ir2[start: end])
            # vmded_red2, red2_disturb = self.preprocess(red2[start: end])
            # R = self.get_R(vmded_ir2, vmded_red2, ir2_disturb, red2_disturb)

            # 小波
            dwted_ir2 = filter_func(ir2[start: end])
            dwted_red2 = filter_func(red2[start: end])
            R = get_R(dwted_ir2, dwted_red2, ir2, red2)

            print("R = ", R)

            res.append([R, spo2_value])
            start += step
            end = start + window
    return res

def main():
    parent_dir = r'E:\my_projects\PPG\data\spo2_compare_test'
    save_path = r'E:\my_projects\PPG\results\R_value_new\results.xlsx'
    results = get_R_list(parent_dir)
    save_to_excel(save_path, results)


if __name__ == '__main__':

    main()