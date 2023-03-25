# -*- coding: utf-8 -*-

"""
@author: Mark Wu
@file: ReadRename.py
@time: 2023/3/1 17:20
"""

import time
import os
import shutil


def copy_rename(old_path=None, new_dir_path=None, remove=False):
    """
    把record文件复制到指定位置
    """
    if old_path is None:
        old_path = r"D:\my_projects_V1\my_projects\采集软件\record.txt"
    if new_dir_path is None:
        new_dir_path = r"D:\my_projects_V1\my_projects\PPG_V1\data\BR\apnea\wu"
    elif not os.path.exists(new_dir_path):
        os.mkdir(new_dir_path)
    file_name = time.strftime("%Y%m%d%H%M%S", time.localtime()) + '.txt'
    save_path = os.path.join(new_dir_path, file_name)
    shutil.copy(old_path, save_path)
    if remove:
        os.remove(old_path)


if __name__ == '__main__':
    # dir_name = "wu"
    # dir_name = "tong"
    # dir_name = "luo"
    # dir_name = "lu"
    # dir_name = "hu"
    # dir_name = "xv"

    # 不同时长
    dir_name = "20"
    # dir_name = "30"
    # dir_name = "40"
    # dir_name = "50"
    # dir_name = "60"


    # root_path = r"D:\my_projects_V1\my_projects\PPG_V1\data\BR\apnea"
    root_path = r"D:\my_projects_V1\my_projects\PPG_V1\data\BR\short_apnea\wu"
    new_path = os.path.join(root_path, dir_name)
    copy_rename(new_dir_path=new_path, remove=True)

