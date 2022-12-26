# @Author: phd
# @Date: 2019-10-22
# @Site: github.com/phdsky
# @Description: NULL

import time
import logging
import numpy as np
import pandas as pd
import random

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Binarizer


def log(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        ret = func(*args, **kwargs)

        end_time = time.time()
        logging.debug('%s() consts %s seconds' % (func.__name__, end_time - start_time))

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


def sign(x):
    if x >= 0:
        return 1
    elif x < 0:
        return -1
    else:
        print("Sign function input wrong!\n")
        exit(-1)


class SVM(object):
    def __init__(self, features, labels, kernelType='linear', C=1.0, epsilon=0.001, tolerance=0.001):
        self.kernelType = kernelType
        self.C = C  # punish parameter
        self.epsilon = epsilon  # slack
        self.tolerance = tolerance  # tolerance

        self.X = features
        self.Y = labels

        self.numOfSamples = len(self.X)

        self.b = 0.
        self.alpha = [0.]*self.numOfSamples
        # self.Ei = [self.calculate_Ei(i) for i in range(self.numOfSamples)]
        self.Ei = [0.]*self.numOfSamples

    def kernel(self, X1, X2):
        X1 = np.asarray(X1)
        X2 = np.asarray(X2)

        if self.kernelType == 'linear':
            return np.inner(X1, X2)
        elif self.kernelType == 'poly':
            self.p = 2  # assumed poly value
            return (np.inner(X1, X2))**self.p
        elif self.kernelType == 'gaussian':
            self.sigma = 10  # assumed sigma value
            return np.exp(-np.inner((X1 - X2), (X1 - X2)) / (2 * self.sigma**2))
        else:
            print("WTF kernel type is?")
            exit(-1)

    def calculate_gxi(self, i):
        return self.b + sum([self.alpha[j]*self.Y[j]*self.kernel(self.X[i], self.X[j])
                             for j in range(self.numOfSamples)])

    def calculate_Ei(self, i):
        return self.calculate_gxi(i) - self.Y[i]

    def is_satisfy_KKT(self, i):
        if (self.alpha[i] == 0) and (self.Y[i]*self.calculate_gxi(i) >= 1. - self.epsilon):
            return True
        elif (0. < self.alpha[i] < self.C) and (np.fabs(self.Y[i]*self.calculate_gxi(i) - 1.) <= self.epsilon):
            return True
        elif (self.alpha[i] == self.C) and (self.Y[i]*self.calculate_gxi(i) <= 1. + self.epsilon):
            return True

        return False

    def select_two_parameters(self):
        # First, select all 0 < alpha < C sample points check these points satisfy KKT or not
        # If all these points(0 < alpha < C) satisfy KKT
        # Then should check all sample points whether satisfy KKT
        # Select one that breaks KKT, and another one has max |E1 - E2| value

        allPointsIndex = [i for i in range(self.numOfSamples)]
        conditionPointsIndex = list(filter(lambda c: 0 < self.alpha[c] < self.C, allPointsIndex))

        unConditionPointsIndex = list(set(allPointsIndex) - set(conditionPointsIndex))
        reArrangePointsIndex = conditionPointsIndex + unConditionPointsIndex

        for i in reArrangePointsIndex:
            if self.is_satisfy_KKT(i):
                continue

            maxIndexEi = (0, 0.)  # (key, value)
            E1 = self.Ei[i]
            for j in allPointsIndex:
                if i == j:
                    continue
                E2 = self.Ei[j]
                if np.fabs(E1 - E2) > maxIndexEi[1]:
                    maxIndexEi = (j, np.fabs(E1 - E2))
            return i, maxIndexEi[0]

        return 0, 0

    def select_i2(self, i1, E1):
        E2 = 0
        i2 = -1
        max_E1_E2 = -1

        non_zero_Ei = [ei for ei in range(self.numOfSamples) if self.calculate_Ei(ei) != 0]
        for e in non_zero_Ei:
            E2_tmp = self.calculate_Ei(e)

            if np.fabs(E1 - E2_tmp) > max_E1_E2:
                max_E1_E2 = np.fabs(E1 - E2_tmp)
                E2 = E2_tmp
                i2 = e

        if i2 == -1:
            i2 = i1
            while i2 == i1:
                i2 = int(random.uniform(0, self.numOfSamples))
            E2 = self.calculate_Ei(i2)
            # E2 = self.Ei[i2]

        return i2, E2

    def smo_trunk(self, i1):
        E1 = self.calculate_Ei(i1)
        # E1 = self.Ei[i1]

        if not self.is_satisfy_KKT(i1):

            i2, E2 = self.select_i2(i1, E1)
            print(i1, i2)

            alpha_i1_old = self.alpha[i1]
            alpha_i2_old = self.alpha[i2]

            if self.Y[i1] != self.Y[i2]:
                L = np.fmax(0., alpha_i2_old - alpha_i1_old)
                H = np.fmin(self.C, self.C + alpha_i2_old - alpha_i1_old)
            elif self.Y[i1] == self.Y[i2]:
                L = np.fmax(0., alpha_i2_old + alpha_i1_old - self.C)
                H = np.fmin(self.C, alpha_i2_old + alpha_i1_old)
            else:
                print("WTF of this condition?")
                exit(-1)

            if L == H:
                return 0

            eta = (self.kernel(self.X[i1], self.X[i1]) + self.kernel(self.X[i2], self.X[i2])) - \
                  (self.kernel(self.X[i1], self.X[i2]) * 2.)

            if eta <= 0:
                return 0

            alpha2_new_unclipped = alpha_i2_old + (self.Y[i2]*(E1 - E2) / eta)

            if alpha2_new_unclipped >= H:
                alpha2_new_clipped = H
            elif L < alpha2_new_unclipped < H:
                alpha2_new_clipped = alpha2_new_unclipped
            elif alpha2_new_unclipped <= L:
                alpha2_new_clipped = L
            else:
                print("WTF of the alpha2_new_uncliped value?")
                print(i1, i2, alpha2_new_unclipped, eta)
                exit(-1)

            if np.fabs(alpha2_new_clipped - alpha_i2_old) < self.tolerance:
                return 0

            s = self.Y[i1]*self.Y[i2]
            alpha1_new = alpha_i1_old + s*(alpha_i2_old - alpha2_new_clipped)

            b1 = - E1 \
                 - self.Y[i1]*self.kernel(self.X[i1], self.X[i1])*(alpha1_new - alpha_i1_old) \
                 - self.Y[i2]*self.kernel(self.X[i2], self.X[i1])*(alpha2_new_clipped - alpha_i2_old)\
                 + self.b
            b2 = - E2 \
                 - self.Y[i1]*self.kernel(self.X[i1], self.X[i2])*(alpha1_new - alpha_i1_old) \
                 - self.Y[i2]*self.kernel(self.X[i2], self.X[i2])*(alpha2_new_clipped - alpha_i2_old) \
                 + self.b

            if 0 < alpha1_new < self.C:
                b = b1
            elif 0 < alpha2_new_clipped < self.C:
                b = b2
            else:
                b = (b1 + b2) / 2.

            self.b = b

            self.alpha[i1] = alpha1_new
            self.alpha[i2] = alpha2_new_clipped

            # Update all error cache Ei value
            self.Ei = [self.calculate_Ei(i) for i in range(self.numOfSamples)]
            # self.Ei[i1] = self.calculate_Ei(i1)
            # self.Ei[i2] = self.calculate_Ei(i2)

            return 1
        else:
            return 0

    def check_not_bound(self):
        return [nb for nb in range(self.numOfSamples) if 0 < self.alpha[nb] < self.C]

    @log
    def train(self, maxIteration=50):
        iterNum = 0
        iterEntireSet = True
        alphaPairsChanged = 0

        while (iterNum < maxIteration) and (alphaPairsChanged > 0 or iterEntireSet):
            iterNum += 1
            print("Iteration: %d of %d" % (iterNum, maxIteration))

            alphaPairsChanged = 0
            if iterEntireSet:
                for i in range(self.numOfSamples):
                    alphaPairsChanged += self.smo_trunk(i)
            else:
                not_bound_list = self.check_not_bound()
                for i in not_bound_list:
                    alphaPairsChanged += self.smo_trunk(i)

            if iterEntireSet:
                iterEntireSet = False
            else:
                iterEntireSet = True

    @log
    def predict(self, X_test):
        n = len(X_test)
        predict_label = np.full(n, -2)

        for i in range(0, n):
            to_predict = X_test[i]

            result = self.b

            for j in range(self.numOfSamples):
                result += self.alpha[j]*self.Y[j]*self.kernel(to_predict, self.X[j])

            predict_label[i] = sign(result)

        return predict_label

    def visualize(self):
        # Extract positive & negative samples
        # conditionPointsIndex = list(filter(lambda c: 0 < self.alpha[c] < self.C, allPointsIndex))
        positive_index = [pos for pos in range(self.numOfSamples) if self.Y[pos] ==  1]
        negative_index = [neg for neg in range(self.numOfSamples) if self.Y[neg] == -1]

        plt.xlabel('X1')  # -12 ~ 12
        plt.ylabel('X2')  # -6 ~ 6

        positive = np.asarray(list(self.X[pos] for pos in positive_index))
        negative = np.asarray(list(self.X[neg] for neg in negative_index))

        plt.scatter(positive[:, 0], positive[:, 1], c='b', s=20)
        plt.scatter(negative[:, 0], negative[:, 1], c='y', s=20)

        for i, text in enumerate(np.arange(self.numOfSamples)):
            plt.annotate(text, (self.X[i][0], self.X[i][1]))

        support_index = [sv for sv in range(self.numOfSamples) if self.alpha[sv] != 0]

        support_alpha = np.asarray(list(self.alpha[sv] for sv in support_index))
        support_vector = np.asarray(list(self.X[sv] for sv in support_index))
        support_label = np.asarray(list(self.Y[sv] for sv in support_index))

        plt.scatter(support_vector[:, 0], support_vector[:, 1], c='', s=100, alpha=0.5, linewidths=1.5, edgecolor='red')
        print("Support Vector Number: ", len(support_vector))
        print(support_index)

        X1 = np.arange(np.min([self.X[x][0] for x in range(self.numOfSamples)]),
                       np.max([self.X[x][0] for x in range(self.numOfSamples)]),
                       0.1)
        X2 = np.arange(np.min([self.X[x][1] for x in range(self.numOfSamples)]),
                       np.max([self.X[x][1] for x in range(self.numOfSamples)]),
                       0.1)

        x1, x2 = np.meshgrid(X1, X2)
        g = self.b

        for i in range(len(support_vector)):
            if self.kernelType == 'linear':
                g += support_alpha[i]*support_label[i]*(x1*support_vector[i][0] + x2*support_vector[i][1])
            elif self.kernelType == 'poly':
                print("Not Implement Yet...")
            elif self.kernelType == 'gaussian':
                g += support_alpha[i]*support_label[i]*\
                     np.exp(-0.5*((x1 - support_vector[i][0])**2 + (x2 - support_vector[i][1])**2) / (self.sigma**2))
            else:
                print("WTF kernel type is?")
                exit(-1)

        plt.contour(x1, x2, g, 0, colors='c')

        plt.show()
        print("Figure plot!")


def load_dataset(filename):
    data = []
    label = []
    with open(filename) as fr:
        for line in fr.readlines():
            line = line.strip().split(' ')
            data.append([float(line[0]), float(line[1])])
            label.append(float(line[2]))
    return data, label


if __name__ == "__main__":
    # logger = logging.getLogger()
    # logger.setLevel(logging.DEBUG)

    # mnist_data = pd.read_csv("../data/mnist_binary.csv")
    # mnist_values = mnist_data.values
    #
    # sample_num = 2000
    # images = mnist_values[:sample_num, 1::]
    # labels = mnist_values[:sample_num, 0]
    #
    # X_train, X_test, y_train, y_test = train_test_split(
    #     images, labels, test_size=0.33, random_state=42
    # )

    X_train, y_train = load_dataset('./dataset.txt')

    svm = SVM(features=X_train, labels=y_train, kernelType='linear')

    print("SVM training...")
    svm.train()
    print("\nTraining done...")

    svm.visualize()
    print(np.nonzero(svm.alpha), svm.b)

    # print("Testing on %d samples..." % len(X_test))
    # y_predicted = svm.predict(X_test=X_test)

    # calc_accuracy(y_pred=y_predicted, y_truth=y_test)