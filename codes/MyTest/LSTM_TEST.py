# -*- coding: utf-8 -*-

"""
@author: Mark Wu
@file: LSTM_TEST.py
@time: 2023/3/8 9:37
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout
from tensorflow.python.keras.layers import LSTM
from tensorflow import keras
# from tensorflow.python.keras.utils import to_categorical


def load_file(filepath):
    dataframe = pd.read_csv(filepath, sep=",")
    # print(dataframe.shape)
    # X = dataframe.iloc[:, 1:]
    # print(X.shape)
    return dataframe.values


def load_dataset(data_rootdir, dir_name):
    '''
    该函数实现将训练数据或测试数据文件列表堆叠为三维数组
    '''

    # filepath_list = []
    X = []
    filepath_list = os.listdir(data_rootdir)
    sorted(filepath_list, key=lambda x: int(os.path.split(x)[1].split(".")[0]))

    for filepath in filepath_list:
        X.append(load_file(os.path.join(data_rootdir, filepath)))

    # X = np.dstack(X)  # dstack沿第三个维度叠加，两个二维数组叠加后，前两个维度尺寸不变，第三个维度增加；
    X = np.array(X)
    file_name = dir_name.split("_")[0]
    train_y_path = os.path.join(os.path.join(os.path.split(data_rootdir)[0], "%s_y" % file_name), "%s_label.csv" % file_name)
    df_y = pd.read_csv(train_y_path, sep=",")
    y = df_y.iloc[:, 1]
    y = np.array(y).reshape(len(X), -1)
    print(y.shape)

    # one-hot编码。这个之前的文章中提到了，因为原数据集标签从1开始，而one-hot编码从0开始，所以要先减去1
    # y = to_categorical(y - 1)
    # print('{}_X.shape:{},{}_y.shape:{}\n'.format(group, X.shape, group, y.shape))
    return X, y


def evaluate_model(trainX, trainy, testX, testy):
    verbose, epochs, batch_size = 0, 15, 64
    print(trainX.shape)
    n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]

    model = Sequential()
    model.add(LSTM(100, input_shape=(n_timesteps, n_features)))
    # model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    # model.add(Dense(n_outputs, activation='softmax'))
    print("suma----", model.summary())
    # adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=1)

    _, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=1)
    return accuracy


def run_experiment(trainX, trainy, testX, testy, repeats=2):
    scores = list()
    for r in range(repeats):
        score = evaluate_model(trainX, trainy, testX, testy)
        score = score * 100.0
        print('>#%d: %.3f' % (r + 1, score))
        scores.append(score)

    m, s = np.mean(scores), np.std(scores)
    print('Accuracy: %.3f%% (+/-%.3f)' % (m, s))


if __name__ == '__main__':
    # train_dir = 'D:/GraduationCode/01 Datasets/UCI HAR Dataset/train/'
    train_dir = r"D:\my_projects_V1\my_projects\PPG_V1\data\BR\LSTMDataset\train_x"
    # test_dir = 'D:/GraduationCode/01 Datasets/UCI HAR Dataset/test/'
    test_dir = r"D:\my_projects_V1\my_projects\PPG_V1\data\BR\LSTMDataset\test_x"

    # dirname = '/Inertial Signals/'
    trainX, trainy = load_dataset(train_dir, "train_x")
    testX, testy = load_dataset(test_dir, "test_x")
    # print(testy)

    run_experiment(trainX, trainy, testX, testy, repeats=1)
