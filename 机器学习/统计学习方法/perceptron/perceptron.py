# @Author: phd
# @Date: 19-3-28
# @Site: github.com/phdsky
# @Description: NULL

import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

## Parameters setting
w0 = 0
b0 = 0
epoch_times = 300

# learning params
rate = 1  # (0 < rate <= 1)
ratio = 3
period = 30


def sign(x):
    if x >= 0:
        return 1
    elif x < 0:
        return -1
    else:
        print("Sign function input wrong!\n")


class Perceptron(object):
    def __init__(self, w, b, epoch, learning_rate, learning_ratio, learning_period):
        self.weight    = w
        self.bias      = b
        self.epoch     = epoch
        self.lr_rate   = learning_rate
        self.lr_ratio  = learning_ratio
        self.lr_period = learning_period

    def train(self, X, y):
        print("Training on train dataset...")

        # Feature arrange - Simple init - Column vector
        self.weight = np.full((len(X[0]), 1), self.weight, dtype=float)

        for epoch in range(self.epoch):
            miss_count = 0
            data_count = len(X)

            for i in range(data_count):  # Single batch training
                feature = X[i]
                label = y[i]

                result = label * (np.dot(feature, self.weight) + self.bias)

                if result <= 0:
                    miss_count += 1
                    self.weight += np.reshape(self.lr_rate * label * feature, (len(feature), 1))
                    self.bias += self.lr_rate * label

            # Print training log
            print("\rEpoch %d\tLearning rate %f\tTraining accuracy %f" %
                  ((epoch + 1), self.lr_rate, (int(data_count - miss_count) / data_count)), end='')

            # Decay learning rate
            if epoch % self.lr_period == 0:
                self.lr_rate /= self.lr_ratio

            # Stop training
            if self.lr_rate <= 1e-6:
                print("Learning rate is too low, Early stopping...\n")
                break

        # print("Parameters after learning")
        # print(self.weight)
        # print(self.bias)
        print("\nEnd of training progress\n")

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
    mnist_data = pd.read_csv("../data/mnist_binary.csv")
    mnist_values = mnist_data.values

    images = mnist_values[::, 1::]
    labels = mnist_values[::, 0]

    X_train, X_test, y_train, y_test = train_test_split(
        images, labels, test_size=0.33, random_state=42
    )

    # bpc - Binary Perceptron Classification
    bpc = Perceptron(w=w0, b=b0, epoch=epoch_times, learning_rate=rate,
                     learning_ratio=ratio, learning_period=period)

    # Start training
    bpc.train(X=X_train, y=y_train)

    # Start predicting
    bpc.predict(X=X_test, y=y_test)
