# @Author: phd
# @Date: 2019/8/19
# @Site: github.com/phdsky
# @Description: NULL

import time
import logging
import numpy as np
import pandas as pd

from collections import defaultdict
from sklearn.model_selection import train_test_split


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


class maxEnt(object):

    def init_params(self, X_train, y_train):
        assert(len(X_train) == len(y_train))

        self.labels = set()
        self.cal_Vxy(X_train, y_train)

        self.N = len(X_train)  # Training set number
        self.n = len(self.Vxy)  # Feature counts
        self.M = 10000.0  # A constant value depends on training set
        self.iter = 500

        self.build_dict()
        self.cal_Pxy()  # Equals to Ep~fi

    def cal_Vxy(self, X_train, y_train):
        # defaultdict: Do not need to judge whether key is in dict or not
        self.Vxy = defaultdict(int)

        # Count the V(X=x, Y=y) feature counts in all samples
        for i in range(0, len(y_train)):
            sample = X_train[i]
            label = y_train[i]

            self.labels.add(label)
            for feature in sample:
                self.Vxy[(feature, label)] += 1

    def build_dict(self):
        # self.Vxy:  key: (x, y)  <---->  value: feature counts
        # Use id key to index

        self.id2xy = {}
        self.xy2id = {}

        for id, xy in enumerate(self.Vxy):
            self.id2xy[id] = xy
            self.xy2id[xy] = id

    def cal_Pxy(self):
        self.Pxy = np.full((self.n, 1), 0.0, dtype=float)

        for id in range(0, self.n):
            xy = self.id2xy[id]
            feature_counts = self.Vxy[xy]

            self.Pxy[id] = feature_counts / float(self.N)

    def cal_Zx(self, sample):
        Zx = defaultdict(float)

        for label in self.labels:
            weights = 0.0
            for feature in sample:
                xy = (feature, label)

                if xy in self.xy2id:
                    id = self.xy2id[xy]
                    weights += self.weight[id]

            Zx[label] = np.exp(weights)

        return Zx

    def cal_Pyx(self, sample):
        Pyx = defaultdict(float)

        Zx = self.cal_Zx(sample)
        Zwx = sum(Zx.values())

        for key in Zx.keys():
            Pyx[key] = Zx[key] / Zwx

        return Pyx

    def cal_Epfi(self, X_train):
        Epfi = np.full((self.n, 1), 0.0, dtype=float)

        for sample in X_train:
            Pyx = self.cal_Pyx(sample)

            for feature in sample:
                for label in Pyx.keys():

                    xy = (feature, label)
                    if xy in self.xy2id:
                        id = self.xy2id[xy]

                        # Calculate P(y|x)*P~(x)f(x, y)
                        # += means every time calculate one to empirical distribution
                        Epfi[id] += Pyx[label] * (1 / self.N)

        return Epfi

    @log
    def train(self, X_train, y_train):
        self.init_params(X_train, y_train)
        self.weight = np.full((self.n, 1), 0.0, dtype=float)

        for it in range(0, self.iter):
            print("Iteration number: %d" % it)
            Epfi = self.cal_Epfi(X_train)

            delta = 1 / self.M * np.log(self.Pxy / Epfi)
            self.weight += delta

    @log
    def predict(self, X_test):
        n = len(X_test)
        predict_label = np.full(n, -1)

        for i in range(0, n):
            to_predict = X_test[i]
            Pyx = self.cal_Pyx(to_predict)
            max_prob = max(zip(Pyx.values(), Pyx.keys()))
            predict_label[i] = max_prob[-1]

        return predict_label


def rebuid_features(subsets):
    features = []
    for sample in subsets:
        feature = []
        for index, value in enumerate(sample):
            feature.append(str(index) + '_' + str(value))
        features.append(feature)
    return features


if __name__ == "__main__":
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    mnist_data = pd.read_csv("../data/mnist.csv")
    mnist_values = mnist_data.values

    sample_num = 5000
    images = mnist_values[:sample_num, 1::]
    labels = mnist_values[:sample_num, 0]

    X_train, X_test, y_train, y_test = train_test_split(
        images, labels, test_size=0.33, random_state=42
    )

    X_train = rebuid_features(subsets=X_train)
    X_test = rebuid_features(subsets=X_test)

    max_ent = maxEnt()

    print("Training max entropy model...")
    max_ent.train(X_train=X_train, y_train=y_train)
    print("Training done...")

    print("Testing on %d samples..." % len(X_test))
    y_predicted = max_ent.predict(X_test=X_test)

    calc_accuracy(y_pred=y_predicted, y_truth=y_test)
