# @Author: phd
# @Date: 2019-03-30
# @Site: github.com/phdsky
# @Description: NULL

import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

## Parameters setting
w0 = 0
b0 = 0
epoch_times = 1

# learning params
rate = 1  # (0 < rate <= 1)
ratio = 3
period = 10


def sign(x):
    if x >= 0:
        return 1
    elif x < 0:
        return -1
    else:
        print("Sign function input wrong!\n")


class PerceptronDual(object):
    def __init__(self, w, b, epoch, learning_rate, learning_ratio, learning_period):
        self.weight    = w
        self.bias      = b
        self.epoch     = epoch
        self.lr_rate   = learning_rate
        self.lr_ratio  = learning_ratio
        self.lr_period = learning_period

    def train(self, X, y):
        print("Training on train dataset...")

        data_count = len(X)

        gram = np.zeros((data_count, data_count))
        # Compute gram matrix - A symmetric matrix
        # Too slow to compute
        for i in range(data_count):
            for j in range(i, data_count):
                gram[i][j] = np.dot(X[i], X[j])
                if i != j:
                    gram[j][i] = gram[i][j]

        alpha = np.zeros((data_count, 1), dtype=float)

        for epoch in range(self.epoch):
            print("--------Epoch %d--------" % (epoch + 1))
            print("Learning rate %f" % self.lr_rate)

            miss_count = 0
            data_arrange = [0, 2, 2, 2, 0, 2, 2]
            for i in data_arrange:
            # for i in range(data_count):
                label_i = y[i]
                result = 0
                for j in range(data_count):
                    label_j = y[j]
                    result += alpha[j] * label_j * gram[i][j]

                result = label_i * (result + self.bias)

                if result <= 0:
                    miss_count += 1
                    alpha[i] += self.lr_rate
                    self.bias += self.lr_rate * label_i

                print("a:", alpha.transpose())
                print("b:", self.bias)
                print("----------------------")

            # print("Training accuracy %f\n" % (int(data_count - miss_count) / data_count))

            # Decay learning rate
            if epoch % self.lr_period == 0:
                self.lr_rate /= self.lr_ratio

            # Stop training
            if self.lr_rate <= 1e-6:
                print("Learning rate is too low, Early stopping...\n")
                break

        alpha_y = alpha * np.reshape(y, alpha.shape)
        self.weight = np.dot(np.transpose(X), alpha_y)

        print("Parameters after learning")
        print("w:", self.weight.transpose())
        print("b:", self.bias)
        print("End of training progress\n")

    def predict(self, X, y):
        print("Predicting on test dataset...")

        hit_count = 0
        data_count = len(X)
        for i in range(data_count):
            feature = X[i]
            label = y[i]

            result = np.dot(feature, self.weight) + self.bias
            predict_label = sign(result)

            if predict_label == label:
                hit_count += 1

        print("Predicting accuracy %f\n" % (int(hit_count) / data_count))


if __name__ == "__main__":
    #mnist_data = pd.read_csv("../data/mnist_binary.csv")
    #mnist_values = mnist_data.values

    #images = mnist_values[::, 1::]
    #labels = mnist_values[::, 0]

    # X_train, X_test, y_train, y_test = train_test_split(
    #     images, labels, test_size=0.33, random_state=42
    # )

    X_train = np.asarray([[3, 3], [4, 3], [1, 1]])
    y_train = np.asarray([1, 1, -1])

    # bdpc - Binary Dual Perceptron Classification
    bdpc = PerceptronDual(w=w0, b=b0, epoch=epoch_times, learning_rate=rate,
                          learning_ratio=ratio, learning_period=period)

    # Start training
    bdpc.train(X=X_train, y=y_train)

    # Start predicting
    #bdpc.predict(X=X_test, y=y_test)
