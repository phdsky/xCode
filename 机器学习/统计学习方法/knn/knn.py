# @Author: phd
# @Date: 19-4-17
# @Site: github.com/phdsky
# @Description:
#   KNN has no explict training progress
#       can deal with multi label classification

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split


def calc_accuracy(y_pred, y_truth):
    assert len(y_pred) == len(y_truth)
    n = len(y_pred)

    hit_count = 0
    for i in range(0, n):
        if y_pred[i] == y_truth[i]:
            hit_count += 1

    print("Predicting accuracy %f\n" % (hit_count / n))


def minkowski(xi, xj, p):
    assert len(xi) == len(xj)
    n = len(xi)

    # distance = 0
    # for i in range(0, n):
    #     distance += pow(abs(xi[i] - xj[i]), p)
    #
    # distance = pow(distance, 1/p)

    # Euclidean distance
    distance = np.linalg.norm(xi - xj)

    return distance


class KNN(object):
    def __init__(self, k, p):
        self.k = k
        self.p = p

    def vote(self, k_vec):
        assert len(k_vec) == self.k
        flag = np.full(10, 0)  # Ten labels

        for i in range(0, self.k):
            flag[k_vec[i][1]] += 1

        return np.argmax(flag)

    def predict(self, X_train, y_train, X_test):
        n = len(X_test)
        m = len(X_train)
        predict_label = np.full(n, -1)

        for i in range(0, n):
            to_predict = X_test[i]
            distances, distances_label = [], []
            for j in range(0, m):
                to_compare = X_train[j]
                dist = minkowski(to_predict, to_compare, self.p)
                distances.append(dist)

            distances_label = list(zip(distances, y_train))
            distances_label.sort(key=lambda kv: kv[0])

            predict_label[i] = self.vote(distances_label[0:self.k])
            print("Nearest neighbour is %s" % X_train[predict_label[i]])
            print("Sample %d predicted as %d" % (i, predict_label[i]))

        return predict_label


def example():
    print("Start testing on simple dataset...")

    X_train = np.asarray([[2, 3], [5, 4], [9, 6], [4, 7], [8, 1], [7, 2]])
    y_train = np.asarray([0, 1, 2, 3, 4, 5])
    X_test = np.asarray([[3, 5]])

    knn = KNN(k=1, p=2)  # p=2 Euclidean distance
    y_predicted = knn.predict(X_train=X_train, y_train=y_train, X_test=X_test)

    print("Simple testing done...\n")


if __name__ == "__main__":

    example()

    # mnist_data = pd.read_csv("../data/mnist.csv")
    # mnist_values = mnist_data.values
    #
    # images = mnist_values[::, 1::]
    # labels = mnist_values[::, 0]
    #
    # X_train, X_test, y_train, y_test = train_test_split(
    #     images, labels, test_size=100, random_state=42
    # )
    #
    # knn = KNN(k=1, p=2)  # p=2 Euclidean distance
    #
    # # Start predicting, training progress omitted
    # print("Testing on %d samples..." % len(X_test))
    # y_predicted = knn.predict(X_train=X_train, y_train=y_train, X_test=X_test)
    #
    # calc_accuracy(y_pred=y_predicted, y_truth=y_test)
