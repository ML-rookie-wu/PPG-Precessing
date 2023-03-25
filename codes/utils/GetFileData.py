#coding:utf-8
import pandas as pd
import os


def travel_dir(parent_dir):
    """
    遍历目录，获取所有txt文件
    :param parent_dir: 父文件夹路径
    :return: 文件路径列表
    """
    all_files = []
    for root_path, dir_names, filenames in os.walk(parent_dir):
        # print(dir_names)
        for file_name in filenames:
            if file_name.find("txt") < 0 or file_name.find("copy") > 0:
                continue
            file_path = os.path.join(root_path, file_name)
            all_files.append(file_path)
    return all_files


def read_from_file(file_path, resp=False):
    data = pd.read_table(file_path, sep=",")
    if data.shape[1] == 6:
        data.columns = ["time", "ir1", "red1", "ir2", "red2", "resp"]
    elif data.shape[1] == 5:
        data.columns = ["ir1", "red1", "ir2", "red2", "resp"]
    elif data.shape[1] == 4:
        data.columns = ["ir1", "red1", "ir2", "red2"]
    else:
        raise Exception("The format of data is wrong")
    # if not resp:
    #     return data.ir2, data.red2
    # else:
    #     return data.ir2, data.red2, resp
    return data

def get_ir2(path):
    data = read_from_file(path)
    ir2 = data.ir2
    return ir2


if __name__ == '__main__':
    parent_dir = r'E:\my_projects\PPG\data\spo2_compare'
    all_files = travel_dir(parent_dir)
    print(all_files)