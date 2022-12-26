# @Author: phd
# @Date: 2019-08-18
# @Site: github.com/phdsky
# @Description: NULL

import time
import logging
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Binarizer


def log(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        ret = func(*args, **kwargs)

        end_time = time.time()
        logging.debug('%s() cost %s seconds' % (func.__name__, end_time - start_time))

        return ret
    return wrapper


def calc_accuracy(y_pred, y_truth):
    assert len(y_pred) == len(y_truth)
    n = len(y_pred)

    hit_count = 0
    for i in range(0, n):
        if y_pred[i] == y_truth[i]:
            hit_count += 1

    print("Predicting accuracy %f" % (hit_count / n))


class LogisticRegression(object):

    def __init__(self, w, b, learning_rate, max_epoch, learning_period, learning_ratio):
        self.weight = w
        self.bias = b
        self.lr_rate = learning_rate
        self.max_epoch = max_epoch
        self.lr_period = learning_period
        self.lr_ratio = learning_ratio

    def calculate(self, feature):
        # wx = sum([self.weight[j] * feature[j] for j in range(len(self.weight))])
        wx = np.dot(self.weight.transpose(), feature)
        exp_wx = np.exp(wx)

        predicted = 0 if (1 / (1 + exp_wx)) > 0.5 else 1

        return predicted, exp_wx

    @log
    def train(self, X_train, y_train):
        # Fuse weight with bias
        self.weight = np.full((len(X_train[0]), 1), self.weight, dtype=float)
        self.weight = np.row_stack((self.weight, self.bias))

        epoch = 0

        while epoch < self.max_epoch:
            hit_count = 0
            data_count = len(X_train)

            for i in range(data_count):
                feature = X_train[i].reshape([len(X_train[i]), 1])
                feature = np.row_stack((feature, 1))

                label = y_train[i]

                predicted, exp_wx = self.calculate(feature)

                if predicted == label:
                    hit_count += 1
                    continue

                # for k in range(len(self.weight)):
                #     self.weight[k] += self.lr_rate * (label*feature[k] - ((feature[k] * exp_wx) / (1 + exp_wx)))
                self.weight += self.lr_rate * feature * (label - (exp_wx / (1 + exp_wx)))

            epoch += 1
            print("\rEpoch %d, lr_rate=%f, Acc = %f" % (epoch, self.lr_rate, hit_count / data_count), end='')

            # Decay learning rate
            if epoch % self.lr_period == 0:
                self.lr_rate /= self.lr_ratio

            # Stop training
            if self.lr_rate <= 1e-6:
                print("\nLearning rate is too low, Early stopping...\n")
                break

    @log
    def predict(self, X_test):
        n = len(X_test)
        predict_label = np.full(n, -1)

        for i in range(0, n):
            to_predict = X_test[i].reshape([len(X_test[i]), 1])
            vec_predict = np.row_stack((to_predict, 1))
            predict_label[i], _ = self.calculate(vec_predict)

        return predict_label


if __name__ == "__main__":
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    mnist_data = pd.read_csv("../data/mnist_binary.csv")
    mnist_values = mnist_data.values

    images = mnist_values[::, 1::]
    labels = mnist_values[::, 0]

    X_train, X_test, y_train, y_test = train_test_split(
        images, labels, test_size=0.33, random_state=42
    )

    # Handle all -1 in y_train to 0
    y_train = y_train * (y_train == 1)
    y_test = y_test * (y_test == 1)

    # Binary the image to avoid predict_probability gets 0
    binarizer_train = Binarizer(threshold=127).fit(X_train)
    X_train_binary = binarizer_train.transform(X_train)

    binarizer_test = Binarizer(threshold=127).fit(X_test)
    X_test_binary = binarizer_test.transform(X_test)

    lr = LogisticRegression(w=0, b=1, learning_rate=0.001, max_epoch=100,
                            learning_period=10, learning_ratio=3)

    print("Logistic regression training...")
    lr.train(X_train=X_train_binary, y_train=y_train)
    print("\nTraining done...")

    print("Testing on %d samples..." % len(X_test))
    y_predicted = lr.predict(X_test=X_test_binary)

    calc_accuracy(y_pred=y_predicted, y_truth=y_test)
