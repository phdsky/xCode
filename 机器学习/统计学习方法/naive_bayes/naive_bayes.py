# @Author: phd
# @Date: 2019/7/10
# @Site: github.com/phdsky
# @Description: NULL

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Binarizer

def calc_accuracy(y_pred, y_truth):
    assert len(y_pred) == len(y_truth)
    n = len(y_pred)

    hit_count = 0
    for i in range(0, n):
        if y_pred[i] == y_truth[i]:
            hit_count += 1

    print("Predicting accuracy %f\n" % (hit_count / n))


class NaiveBayes(object):
    def __init__(self, _lambda, Sj, K):
        self._lambda = _lambda
        self.Sj = Sj  # Feature Dimension (Simple assume to the same)
        self.K = K  # Label range

    # Use bayes estimate
    # Not max-likelihood estimate, avoid probability is 0
    def train(self, X_train, y_train):
        # Calculate prior probability & conditional probability
        N = len(y_train)
        D = X_train.shape[1]  # Dimension

        prior = np.full(self.K, 0)
        condition = np.full((self.K, D, self.Sj), 0)
        # conditional_probability = np.full((self.K, D, self.Sj), 0.)

        for i in range(0, N):
            prior[y_train[i]] += 1
            for j in range(0, D):
                condition[y_train[i]][j][X_train[i][j]] += 1

        prior_probability = (prior + self._lambda) / (N + self.K*self._lambda)

        # Too Slow
        # for i in range(0, self.K):
        #     for j in range(0, D):
        #         for k in range(0, self.Sj):
        #             conditional_probability[i][j][k] = \
        #                 (condition[i][j][k] + self._lambda) / (sum(condition[i][j]) + self.Sj*self._lambda)

        return prior_probability, condition  # , conditional_probability

    def predict(self, prior_probability, condition, X_test):
        n = len(X_test)
        d = X_test.shape[1]

        predict_label = np.full(n, -1)

        for i in range(0, n):
            predict_probability = np.full(self.K, 1.)
            to_predict = X_test[i]

            for j in range(0, self.K):
                prior_prob = prior_probability[j]

                # If d or self.Sj is large, predict_probability gets close to 0
                for k in range(0, d):
                    conditional_probability = \
                        (condition[j][k][to_predict[k]] + self._lambda) / (sum(condition[j][k]) + self.Sj*self._lambda)
                    predict_probability[j] *= conditional_probability

                predict_probability[j] *= prior_prob

            predict_label[i] = np.argmax(predict_probability)

            print("Sample %d predicted as %d" % (i, predict_label[i]))

        return predict_label


def example_large():
    mnist_data = pd.read_csv("../data/mnist.csv")
    mnist_values = mnist_data.values

    images = mnist_values[::, 1::]
    labels = mnist_values[::, 0]

    X_train, X_test, y_train, y_test = train_test_split(
        images, labels, test_size=100, random_state=42
    )

    # Binary the image to avoid predict_probability gets 0
    binarizer_train = Binarizer(threshold=127).fit(X_train)
    X_train_binary = binarizer_train.transform(X_train)

    binarizer_test = Binarizer(threshold=127).fit(X_test)
    X_test_binary = binarizer_test.transform(X_test)

    # Laplace Smoothing
    # X values 0~255 = 256 Every axis has the same range
    # Y values 0~9 = 10
    naive_bayes = NaiveBayes(_lambda=1, Sj=2, K=10)

    print("Start naive bayes training...")
    prior, conditional = naive_bayes.train(X_train=X_train_binary, y_train=y_train)

    print("Testing on %d samples..." % len(X_test))
    y_predicted = naive_bayes.predict(prior_probability=prior,
                                      condition=conditional,
                                      X_test=X_test_binary)

    calc_accuracy(y_pred=y_predicted, y_truth=y_test)


def example_small():
    X_train = np.asarray([[0, 0], [0, 1], [0, 1], [0, 0], [0, 0],
                          [1, 0], [1, 1], [1, 1], [1, 2], [1, 2],
                          [2, 2], [2, 1], [2, 1], [2, 2], [2, 2]])

    y_train = np.asarray([0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0])

    X_test = np.asarray([[1, 0]])

    naive_bayes = NaiveBayes(_lambda=1, Sj=3, K=2)

    print("Start naive bayes training...")
    prior, conditional = naive_bayes.train(X_train=X_train, y_train=y_train)

    print("Testing on %d samples..." % len(X_test))
    naive_bayes.predict(prior_probability=prior,
                        condition=conditional,
                        X_test=X_test)


if __name__ == "__main__":
    # example_small()
    example_large()
