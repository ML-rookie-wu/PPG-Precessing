#coding:utf-8
from codes.utils.GetFileData import travel_dir, read_from_file
from codes.utils.GetFFT import get_freq, signal_fft
from codes.utils.MyFilters import bandpass_filter
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path


def myplot(*args):
    num = len(args)
    fig = plt.figure(figsize=(10, 8))
    for index in range(num):
        ax = fig.add_subplot(num + 1, 1, index+1)
        ax.set_title("fig-%s" % (index + 1))
        if isinstance(args[index], tuple):
            ax.plot(args[index][0], args[index][1])
        else:
            ax.plot(args[index])
        plt.subplots_adjust(hspace=0.8)
    plt.show()

def save_to_excel(data, path):
    writer = pd.ExcelWriter(path, engine="openpyxl")
    df = pd.DataFrame(data)
    df.columns = ['disturb', 'pulse', 'pulse_amplitude', 'resp_amplitude']
    df.to_excel(writer, index=False)
    writer.save()

def PulseTest():
    dir = r'E:\my_projects\PPG\data\pulse_compare'
    excel_path = r'E:\my_projects\PPG\results\pulse\pulse_compare_4000.xlsx'
    all_path = travel_dir(dir)
    result = []
    for path in all_path:
        print(path)
        pulse = Path(path).parent.name
        disturb = Path(Path(path).parent).parent.name[0:1]
        data = read_from_file(path)
        ir2 = data.ir2[0:4000]
        buttered1 = bandpass_filter(ir2, start_fs=0.1, end_fs=0.7)
        buttered2 = bandpass_filter(ir2, start_fs=0.7, end_fs=2)

        f1, absY1 = signal_fft(buttered1)
        max_freq1, max_ap1 = get_freq(f1, absY1)

        f2, absY2 = signal_fft(buttered2)
        max_freq2, max_ap2 = get_freq(f2, absY2)

        # print((max_freq1, max_ap1), (max_freq2, max_ap2))
        result.append([disturb, pulse, round(max_ap2, 3), round(max_ap1, 3)])
        # myplot(ir2, (f1, absY1), (f2, absY2))
    save_to_excel(result, excel_path)


if __name__ == '__main__':
    PulseTest()

