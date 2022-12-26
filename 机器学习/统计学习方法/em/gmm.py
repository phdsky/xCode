# @Author: phd
# @Date: 2019/11/11
# @Site: github.com/phdsky
# @Description: NULL

import time
import logging

import numpy as np
from scipy.stats import multivariate_normal

from matplotlib import pyplot as plt
from sklearn.datasets.samples_generator import make_blobs


class GMM(object):
    def __init__(self, K, X, init_method):
        self.K = K
        self.Y = X
        self.N, self.D = X.shape  # data shape

        self.init = init_method
        if self.init == 'random':
            self.mean = np.random.rand(self.K, self.D)
            self.cov = np.asarray([np.eye(self.D)] * K)
            self.alpha = np.asarray([1. / self.K] * self.K)

        elif self.init == 'kmeans':
            print("Not implemented yet...")
        else:
            print("WTF the init type is?")

    def calc_phi(self, Y, mean, cov):
        return multivariate_normal.pdf(x=Y, mean=mean, cov=cov)

    def E_step(self):
        gamma = np.zeros((self.N, self.K))

        for k in range(self.K):
            gamma[:, k] =  self.calc_phi(self.Y, self.mean[k], self.cov[k])

        for k in range(self.K):
            gamma[:, k] *= self.alpha[k]

        for n in range(self.N):
            gamma[n, :] /= np.sum(gamma[n, :])

        return gamma

    def M_step(self, gamma):
        # cov computation use old mean value
        # so, update cov first
        for k in range(self.K):
            gamma_k = np.reshape(gamma[:, k], (self.N, 1))
            gamma_k_sum = np.sum(gamma_k)

            # cov
            y_mean = self.Y - self.mean[k]
            self.cov[k] = y_mean.T.dot(np.multiply(y_mean, gamma_k)) / gamma_k_sum

            # mean
            self.mean[k] = np.sum(np.multiply(gamma_k, self.Y), axis=0) / gamma_k_sum

            # alpha
            self.alpha[k] = gamma_k_sum / self.N

    def train(self, max_iteration):
        for i in range(max_iteration):
            print("Iteration: %d" % i)

            # Take E Step
            gamma = self.E_step()

            # Take M Step
            self.M_step(gamma=gamma)

    def predict(self):
        gamma = self.E_step()

        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
        predictions = np.argmax(gamma, axis=1)
        for k in range(self.K):

            # plot mean point
            plt.scatter(self.mean[k][0], self.mean[k][1], c='black', edgecolors='none', marker='D')

            # plot points
            pred_ids = np.where(predictions == k)
            plt.scatter(self.Y[pred_ids[0], 0], self.Y[pred_ids[0], 1], c=colors[k], alpha=0.4, edgecolors='none', marker='s')

        plt.show()

if __name__ == "__main__":
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    clusters = 4
    X, y = make_blobs(n_samples=100*clusters, centers=clusters, cluster_std=0.5, random_state=0)
    # plt.scatter(X[:, 0], X[:, 1])
    # plt.show()

    # Available init method: random / kmeans
    gmm = GMM(K=clusters, X=X, init_method='random')

    gmm.train(max_iteration=200)

    gmm.predict()