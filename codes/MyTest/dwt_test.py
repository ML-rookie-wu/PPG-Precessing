#coding:utf-8
from codes.utils.MyFilters import bandpass_filter
from codes.utils.GetFileData import read_from_file, travel_dir
import matplotlib.pyplot as plt
import pywt


def get_dwt_res(data, w='db33', n=10):
    """Decompose and plot a signal S.
    S = An + Dn + Dn-1 + ... + D1
    """
    mode = pywt.Modes.smooth
    w = pywt.Wavelet(w)            # 选取小波函数
    a = data
    ca = []   # 近似分量, a表示低频近似部分
    cd = []   # 细节分量, b表示高频细节部分
    for i in range(n):
        (a, d) = pywt.dwt(a, w, mode)#进行n阶离散小波变换
        ca.append(a)
        cd.append(d)

    rec_a = []
    rec_d = []

    for i, coeff in enumerate(ca):
        coeff_list = [coeff, None] + [None] * i
        rec_a.append(pywt.waverec(coeff_list, w))      #重构

    for i, coeff in enumerate(cd):  # i, coeff 分别对应ca中的下标和元素，分了几层i就为几
        coeff_list = [None, coeff] + [None] * i
        rec_d.append(pywt.waverec(coeff_list, w))

    fig = plt.figure(figsize=(16, 12))
    ax_main = fig.add_subplot(len(rec_a) + 1, 1, 1)
    # ax_main.set_title(title)
    ax_main.plot(data)
    ax_main.set_xlim(0, len(data) - 1)

    for i, y in enumerate(rec_a):
        ax = fig.add_subplot(len(rec_a) + 1, 2, 3 + i * 2)
        ax.plot(y, 'r')
        ax.set_xlim(0, len(y) - 1)
        ax.set_ylabel("A%d" % (i + 1))
        plt.subplots_adjust(hspace=0.5)

    for i, y in enumerate(rec_d):
        ax = fig.add_subplot(len(rec_d) + 1, 2, 4 + i * 2)
        ax.plot(y, 'g')
        ax.set_xlim(0, len(y) - 1)
        ax.set_ylabel("D%d" % (i + 1))
        plt.subplots_adjust(hspace=0.5)

    plt.show()

    return rec_a, rec_d

def dwt(data):
    buttered = bandpass_filter(data, start_fs=0.1, end_fs=5)

    ac, dc = get_dwt_res(buttered)
    resp_disturb = ac[-2][0:len(data)]
    removed = [buttered[i] - resp_disturb[i] for i in range(len(data))]

    return removed, resp_disturb


def MyPlot(*args):
    num = len(args)
    fig = plt.figure(figsize=(10, 8))
    for i in range(1, num+1):
        ax = fig.add_subplot(num, 1, i)
        ax.plot(args[i-1])
        plt.title("图%s" % i)
        plt.subplots_adjust(hspace=0.8)
    plt.show()


def main():
    parent_dir = r'E:\my_projects\PPG\data\test_data\test'
    files = travel_dir(parent_dir)
    for file in files:
        ir2, red2 = read_from_file(file)
        dwted_ir2, resp_disturb = dwt(ir2[2000:6000])
        MyPlot(ir2, dwted_ir2, resp_disturb)
        



if __name__ == '__main__':
    main()





