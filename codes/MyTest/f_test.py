# -*- coding: utf-8 -*-

"""
@author: Mark Wu
@file: f_test.py
@time: 2023/3/25 11:01
"""

import numpy as np
import matplotlib.pyplot as plt
import numpy as np

from scipy.signal import argrelmax, argrelmin, welch
from codes.utils.GetFileData import read_from_file
from codes.utils.MyFilters import bandpass_filter

# from params import PPG_SAMPLE_RATE
# from params import ECG_MF_HRV_CUTOFF, ECG_HF_HRV_CUTOFF


def get_data():
    file_path = r"D:\my_projects_V1\my_projects\PPG_V1\data\respiration\2\20230314200844.txt"
    data = read_from_file(file_path)
    ir = data.ir2
    raw_data = np.array(ir[5000:9000])
    buttered = bandpass_filter(raw_data, start_fs=0.1, end_fs=5, fs=500)

    plt.subplot(211)
    plt.plot(raw_data)
    plt.subplot(212)
    plt.plot(buttered)
    plt.show()

    return buttered


def extract_ppg45(single_waveform, sample_rate=500):
    def __next_pow2(x):
        return 1<<(x-1).bit_length()
    features = []
    maxima_index = argrelmax(np.array(single_waveform))[0]
    minima_index = argrelmin(np.array(single_waveform))[0]
    derivative_1 = np.diff(single_waveform, n=1) * float(sample_rate)
    derivative_1_maxima_index = argrelmax(np.array(derivative_1))[0]
    derivative_1_minima_index = argrelmin(np.array(derivative_1))[0]
    derivative_2 = np.diff(single_waveform, n=2) * float(sample_rate)
    derivative_2_maxima_index = argrelmax(np.array(derivative_2))[0]
    derivative_2_minima_index = argrelmin(np.array(derivative_2))[0]
    sp_mag = np.abs(np.fft.fft(single_waveform, n=__next_pow2(len(single_waveform))*16))
    freqs = np.fft.fftfreq(len(sp_mag))
    sp_mag_maxima_index = argrelmax(sp_mag)[0]
    # x
    x = single_waveform[maxima_index[0]]
    features.append(x)
    # y
    y = single_waveform[maxima_index[1]]
    features.append(y)
    # z
    z = single_waveform[minima_index[0]]
    features.append(z)
    # t_pi
    t_pi = float(len(single_waveform)) / float(sample_rate)
    features.append(t_pi)
    # y/x
    features.append(y / x)
    # (x-y)/x
    features.append((x - y) / x)
    # z/x
    features.append(z / x)
    # (y-z)/x
    features.append((y - z) / x)
    # t_1
    t_1 = float(maxima_index[0] + 1) / float(sample_rate)
    features.append(t_1)
    # t_2
    t_2 = float(minima_index[0] + 1) / float(sample_rate)
    features.append(t_2)
    # t_3
    t_3 = float(maxima_index[1] + 1) / float(sample_rate)
    features.append(t_3)
    # delta_t
    delta_t = t_3 - t_2
    features.append(delta_t)
    # width
    single_waveform_halfmax = max(single_waveform) / 2
    width = 0
    for value in single_waveform[maxima_index[0]::-1]:
        if value >= single_waveform_halfmax:
            width += 1
        else:
            break
    for value in single_waveform[maxima_index[0]+1:]:
        if value >= single_waveform_halfmax:
            width += 1
        else:
            break
    features.append(float(width) / float(sample_rate))
    # A_2/A_1
    features.append(sum(single_waveform[:maxima_index[0]]) / sum(single_waveform[maxima_index[0]:]))
    # t_1/x
    features.append(t_1 / x)
    # y/(t_pi-t_3)
    features.append(y / (t_pi - t_3))
    # t_1/t_pi
    features.append(t_1 / t_pi)
    # t_2/t_pi
    features.append(t_2 / t_pi)
    # t_3/t_pi
    features.append(t_3 / t_pi)
    # delta_t/t_pi
    features.append(delta_t / t_pi)
    # t_a1
    t_a1 = float(derivative_1_maxima_index[0]) / float(sample_rate)
    features.append(t_a1)
    # t_b1
    t_b1 = float(derivative_1_minima_index[0]) / float(sample_rate)
    features.append(t_b1)
    # t_e1
    t_e1 = float(derivative_1_maxima_index[1]) / float(sample_rate)
    features.append(t_e1)
    # t_f1
    t_f1 = float(derivative_1_minima_index[1]) / float(sample_rate)
    features.append(t_f1)
    # b_2/a_2
    a_2 = derivative_2[derivative_2_maxima_index[0]]
    b_2 = derivative_2[derivative_2_minima_index[0]]
    features.append(b_2 / a_2)
    # e_2/a_2
    e_2 = derivative_2[derivative_2_maxima_index[1]]
    features.append(e_2 / a_2)
    # (b_2+e_2)/a_2
    features.append((b_2 + e_2) / a_2)
    # t_a2
    t_a2 = float(derivative_2_maxima_index[0]) / float(sample_rate)
    features.append(t_a2)
    # t_b2
    t_b2 = float(derivative_2_minima_index[0]) / float(sample_rate)
    features.append(t_b2)
    # t_a1/t_pi
    features.append(t_a1 / t_pi)
    # t_b1/t_pi
    features.append(t_b1 / t_pi)
    # t_e1/t_pi
    features.append(t_e1 / t_pi)
    # t_f1/t_pi
    features.append(t_f1 / t_pi)
    # t_a2/t_pi
    features.append(t_a2 / t_pi)
    # t_b2/t_pi
    features.append(t_b2 / t_pi)
    # (t_a1-t_a2)/t_pi
    features.append((t_a1 - t_a2) / t_pi)
    # (t_b1-t_b2)/t_pi
    features.append((t_b1 - t_b2) / t_pi)
    # (t_e1-t_2)/t_pi
    features.append((t_e1 - t_2) / t_pi)
    # (t_f1-t_3)/t_pi
    features.append((t_f1 - t_3) / t_pi)
    # f_base
    f_base = freqs[sp_mag_maxima_index[0]] * sample_rate
    features.append(f_base)
    # sp_mag_base
    sp_mag_base = sp_mag[sp_mag_maxima_index[0]] / len(single_waveform)
    features.append(sp_mag_base)
    # f_2
    f_2 = freqs[sp_mag_maxima_index[1]] * sample_rate
    features.append(f_2)
    # sp_mag_2
    sp_mag_2 = sp_mag[sp_mag_maxima_index[1]] / len(single_waveform)
    features.append(sp_mag_2)
    # f_3
    f_3 = freqs[sp_mag_maxima_index[2]] * sample_rate
    features.append(f_3)
    # sp_mag_3
    sp_mag_3 = sp_mag[sp_mag_maxima_index[2]] / len(single_waveform)
    features.append(sp_mag_3)
    return features


def detect_peaks(signal, neighborhood_size=20):
    """
    Detects peaks in a 1D signal using the dynamic threshold method.
    neighborhood_size表示峰值点邻域的大小，threshhold表示动态阈值的本书
    """
    # Compute the first-order difference of the signal
    diff = np.diff(signal)

    # Initialize the threshold as the median of the first-order differences
    threshold = np.median(np.abs(diff))

    # Initialize an array to store the peak indices
    peaks = []

    # Find peaks by comparing each point to the local threshold
    for i in range(neighborhood_size, len(signal) - neighborhood_size):
        if signal[i] > threshold and signal[i] == np.max(signal[i - neighborhood_size:i + neighborhood_size]):
            peaks.append(i)

        # Update the threshold using the last neighborhood_size peaks
        if len(peaks) >= neighborhood_size:
            threshold = np.median(np.abs(np.diff(signal[peaks[-neighborhood_size:]])))

    return np.array(peaks)


def dynamic_threshold_peak_detection(signal, window_size=10, k=3, alpha=0.2):
    """
    Dynamic threshold peak detection algorithm.
    """
    # Compute the first-order difference of the signal
    diff_signal = np.diff(signal)

    # Initialize the threshold as the median of the first-order differences
    threshold = np.median(np.abs(diff_signal))
    print("thresh = ", threshold)

    # Initialize an empty list to store the peak indices
    peak_indices = []

    # Iterate over the signal with a sliding window
    for i in range(window_size, len(signal) - window_size):
        # Compute the local threshold using k times the standard deviation of the signal
        local_threshold = k * np.std(signal[i - window_size:i + window_size])

        # Check if the absolute difference is above the local threshold and if the value is positive
        if (np.abs(diff_signal[i - 1]) > local_threshold) and (diff_signal[i - 1] > 0):
            # Check if the difference is above the global threshold
            if np.abs(diff_signal[i - 1]) > (1 + alpha) * threshold:
                # Add the index to the peak indices
                peak_indices.append(i - 1)

            # Update the threshold using the previous window_size peaks
            threshold = np.median(np.abs(diff_signal[peak_indices[-window_size:]]))

    return np.array(peak_indices)

def dynamic_threshold_peak_detection1(signal, window_size=10, k=3, alpha=0.2):
    """
    Dynamic threshold peak detection algorithm.
    """
    peak_indices = []

    for i in range(window_size, len(signal) - window_size):
        temp_signal = signal[i - window_size:i + window_size]
        diff_signal = np.diff(temp_signal)
        threshold = np.median(np.abs(temp_signal))
        # Compute the local threshold using k times the standard deviation of the signal
        local_threshold = k * np.std(signal[i - window_size:i + window_size])

        # Check if the absolute difference is above the local threshold and if the value is positive
        if (np.abs(diff_signal[i - 1]) > local_threshold) and (diff_signal[i - 1] > 0):
            # Check if the difference is above the global threshold
            if np.abs(diff_signal[i - 1]) > (1 + alpha) * threshold:
                # Add the index to the peak indices
                peak_indices.append(i - 1)

            # Update the threshold using the previous window_size peaks
            threshold = np.median(np.abs(diff_signal[peak_indices[-window_size:]]))

    return np.array(peak_indices)

def main():
    data = get_data()
    features = extract_ppg45(data)
    peaks = dynamic_threshold_peak_detection1(data, window_size=100, k=1)
    print(peaks)
    # peaks = detect_peaks(data, neighborhood_size=350)
    peak_value = [data[x] for x in peaks]
    plt.plot(data)
    plt.scatter(peaks, peak_value)
    plt.show()
    print(features)




if __name__ == '__main__':
    main()

