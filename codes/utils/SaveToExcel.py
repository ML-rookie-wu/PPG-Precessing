# -*- coding: utf-8 -*-

"""
@author: Mark Wu
@file: SaveToExcel.py
@time: 2023/2/9 13:18
"""
import openpyxl
import pandas as pd
import random


def save_to_excel(save_path, data, column_name):
    writer = pd.ExcelWriter(save_path, engine="openpyxl")
    df = pd.DataFrame(data)
    df.columns = column_name
    df.to_excel(writer, index=False)
    writer.save()

def add_sheet_to_excel(df, excel_path=None, sheet_name=None):

    if excel_path is None:
        excel_path = r'E:\my_projects\PPG\results\errors\errors_all.xlsx'
    wb = openpyxl.load_workbook(excel_path)
    writer = pd.ExcelWriter(excel_path, engine="openpyxl")
    writer.book = wb
    if sheet_name is None:
        sheet_name = "test_%s" % str(random.randint(1, 100))
    df.to_excel(writer, sheet_name=sheet_name, index=False)
    writer.save()