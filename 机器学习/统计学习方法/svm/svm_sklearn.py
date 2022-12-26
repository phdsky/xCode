# @Author: phd
# @Date: 2019-11-05
# @Site: github.com/phdsky
# @Description: NULL

import numpy as np
import pandas as pd

from sklearn import svm
from matplotlib import pyplot as plt


def load_dataset(filename):
    data = []; label = []
    with open(filename) as fr:
        for line in fr.readlines():
            line = line.strip().split(' ')
            data.append([float(line[0]), float(line[1])])
            label.append(float(line[2]))
    return data, label


def plot_point(data, label, support_vector, weight, bias):
    for i in range(np.shape(data)[0]):
        if label[i] == 1:
            plt.scatter(data[i][0], data[i][1], c='b', s=20)
        else:
            plt.scatter(data[i][0], data[i][1], c='y', s=20)

    for j in support_vector:
        plt.scatter(data[j][0], data[j][1], s=100, c='', alpha=0.5, linewidth=1.5, edgecolor='red')

    for i, text in enumerate(np.arange(len(data))):
        plt.annotate(text, (data[i][0], data[i][1]))

    x = np.arange(0, 10, 0.01)
    y = (weight[0][0] * x + bias) / (-1 * weight[0][1])
    plt.scatter(x, y, s=5, marker='h')
    plt.show()


if __name__ == "__main__":
    X_train, y_train = load_dataset('./dataset.txt')

    clf = svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
              decision_function_shape='ovr', degree=3, gamma='auto', kernel='linear',
              max_iter=-1, probability=False, random_state=None, shrinking=True,
              tol=0.001, verbose=False)

    clf.fit(X_train, y_train)

    support_vector_number = clf.n_support_
    support_vector_index = clf.support_

    w = clf.coef_
    b = clf.intercept_
    plot_point(X_train, y_train, support_vector_index, w, b)
    print(w, b)