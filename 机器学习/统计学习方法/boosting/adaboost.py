# @Author: phd
# @Date: 2019-11-08
# @Site: github.com/phdsky
# @Description: NULL

import time
import logging
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Binarizer


def log(func):
    def warpper(*args, **kwargs):
        start_time = time.time()
        ret = func(*args, **kwargs)

        end_time = time.time()
        logging.debug("%s() cost %s seconds" % (func.__name__, end_time - start_time))

        return ret

    return warpper


def calc_accuracy(y_pred, y_truth):
    assert len(y_pred) == len(y_truth)
    n = len(y_pred)

    hit_count = 0
    for i in range(0, n):
        if y_pred[i] == y_truth[i]:
            hit_count += 1

    print("Accuracy %f\n" % (hit_count / n))

    return hit_count / n


def sign(x):
    if x >= 0:
        return 1
    elif x < 0:
        return -1
    else:
        print("Sign function input wrong!\n")


class AdaBoost(object):
    def __init__(self, X_train, y_train, max_classfifers):
        self.X = X_train
        self.Y = y_train

        self.sample_num = len(X_train)  # sample num
        self.feature_num = len(X_train[0])  # feature num
        self.D = np.full(self.sample_num, (1./self.sample_num))  # weight distribution

        self.M = max_classfifers  # max classifier number
        self.axis = np.full(self.M, -1)  # min ei axis selected
        self.alpha = np.zeros(self.M)
        self.Gm = np.zeros(self.M)  # basic classifier

        self.thresh_array = np.arange(np.min(self.X)-0.5, np.max(self.X)+0.51, 1)
        self.direction = np.full(self.M, -1)

    def basic_classifier(self, threshold, value, direction):
        if direction == 0:
            if value < threshold:
                return 1
            else:
                return -1
        elif direction == 1:
            if value > threshold:
                return 1
            else:
                return -1
        else:
            print("WTF the operation direction is?")

    def train_basic_classifier(self, classifier):
        # After binarization, the value is 0 ~ 1, so the
        # threshold should be [-0.5, 0.5, 1.5]
        # For multi dimensional data, choose the axis which
        # has the min ei value to take part in decision
        min_ei = self.sample_num  # all weight is 1 and hit
        selected_axis = -1
        threshold = self.thresh_array[-1] + 1

        direction_array = [0, 1]
        direction = -1

        for axis in range(self.feature_num):
            for th in self.thresh_array:
                axis_vector = self.X[:, axis]
                thresh_vector = np.full(self.sample_num, th)

                for direct in direction_array:
                    # Use vector format calculation for accelerating
                    if direct == 0:
                        compare_vector = np.asarray([axis_vector < thresh_vector], dtype=int) * 2 - 1
                    elif direct == 1:
                        compare_vector = np.asarray([axis_vector > thresh_vector], dtype=int) * 2 - 1

                    calc_ei = np.sum((compare_vector != self.Y)*self.D)

                    # calc_ei = 0.
                    # for sample in range(self.sample_num):
                    #     calc_ei += self.D[sample]*\
                    #                int(self.basic_classifier(thresh, self.X[sample][axis]) != self.Y[sample])

                    if calc_ei < min_ei:
                        min_ei = calc_ei
                        selected_axis = axis
                        threshold = th
                        direction = direct

        self.axis[classifier] = selected_axis
        self.Gm[classifier] = threshold
        self.direction[classifier] = direction

        return min_ei

    @log
    def train(self):
        m = 0
        while m < self.M:
            print("Training %d classifier..." % m)

            # Train basic classifier and classify error
            ei = self.train_basic_classifier(classifier=m)

            # Calculate alpha value
            self.alpha[m] = 0.5*np.log((1 - ei) / ei)

            # Validate training
            train_label = self.predict(X_test=self.X, classifier_number=(m + 1))
            accuracy = calc_accuracy(train_label, self.Y)

            if accuracy == 1.:
                print("Fitting perfect on training set!")
                return m + 1

            # Calculate regulator
            Zm = 0.
            for i in range(self.sample_num):
                Zm += self.D[i] * np.exp(-self.alpha[m]*self.Y[i] *
                                         self.basic_classifier(self.Gm[m], self.X[i][self.axis[m]], self.direction[m]))

            # Update weight distribution
            for i in range(self.sample_num):
                self.D[i] = self.D[i] * np.exp(-self.alpha[m]*self.Y[i] *
                                               self.basic_classifier(self.Gm[m], self.X[i][self.axis[m]], self.direction[m])) / Zm

            m += 1

        return m

    # @log
    def predict(self, X_test, classifier_number):
        n = len(X_test)
        predict_label = np.full(n, -1)

        for i in range(n):
            to_predict = X_test[i]
            result = 0.

            for m in range(classifier_number):
                result += self.alpha[m] * self.basic_classifier(self.Gm[m], to_predict[self.axis[m]], self.direction[m])

            predict_label[i] = sign(result)

        return predict_label


def example_large():
    mnist_data = pd.read_csv("../data/mnist_binary.csv")
    mnist_values = mnist_data.values

    images = mnist_values[::, 1::]
    labels = mnist_values[::, 0]

    X_train, X_test, y_train, y_test = train_test_split(
        images, labels, test_size=0.33, random_state=42
    )

    # Binary the images to avoid AdaBoost classifier threshold complex
    binarizer_train = Binarizer(threshold=127).fit(X_train)
    X_train_binary = binarizer_train.transform(X_train)

    binarizer_test = Binarizer(threshold=127).fit(X_test)
    X_test_binary = binarizer_test.transform(X_test)

    adaboost = AdaBoost(X_train=X_train_binary, y_train=y_train, max_classfifers=233)

    print("AdaBoost training...")
    classifier_trained = adaboost.train()
    print("\nTraining done...")
    print("\nTraining done with %d classifiers!" % classifier_trained)

    print("Testing on %d samples..." % len(X_test))
    y_predicted = adaboost.predict(X_test=X_test_binary, classifier_number=classifier_trained)

    calc_accuracy(y_pred=y_predicted, y_truth=y_test)


def example_small():
    X_train = np.asarray([[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]])
    y_train = np.asarray([1, 1, 1, -1, -1, -1, 1, 1, 1, -1])

    adaboost = AdaBoost(X_train=X_train, y_train=y_train, max_classfifers=5)

    print("Adaboost training...")
    classifier_trained = adaboost.train()
    print("\nTraining done with %d classifiers!" % classifier_trained)


if __name__ == "__main__":
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # example_large()
    example_small()
