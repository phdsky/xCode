# @Author: phd
# @Date: 2019/8/5
# @Site: github.com/phdsky
# @Description: NULL

import numpy as np
import pandas as pd
import math
import time
import logging

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


def empire_entropy(y_dict):
    entropy = 0
    y_sum = sum(y_dict.values())
    for (k, v) in y_dict.items():
        part = v / y_sum
        entropy += part*np.log2(part)  # math.log2(part)

    return -entropy


class TreeNode(object):
    def __init__(self, type=None, belong=None, index=None, subtree=None):
        self.type = type  # Internal or Leaf node type
        self.belong = belong  # Leaf: Belong to which class
        self.index = index  # Internal: Feature index
        self.subtree = subtree  # Internal: Subtree dict


class DecisionTree(object):
    def __init__(self, algorithm, epsilon):
        self.algorithm = algorithm
        self.epsilon = epsilon
        self.root = None

    @log
    def train(self, X_train, y_train):
        feature_indices = list(range(0, len(X_train[0])))
        self.root = self.build(X_train, y_train, feature_indices)

    def build(self, X_set, y_set, indices):
        assert(len(X_set) == len(y_set))

        set_length = len(y_set)

        y_dict = {}
        for i in range(0, set_length):
            if y_set[i] in y_dict.keys():
                y_dict[y_set[i]] += 1
            else:
                y_dict[y_set[i]] = 1

        # Step 1: If all samples belongs to one class, return the node
        if len(y_dict) == 1:
            return TreeNode(type='leaf', belong=y_set[0])

        # Step 2: If indices is empty, vote for the max class node
        if len(indices) == 0:
            return TreeNode(type='leaf', belong=sorted(y_dict, key=lambda x: y_dict[x])[-1])

        # Step 3: Calculate the information gain of all the feature indices
        HD = empire_entropy(y_dict)
        HD_A = []
        H_A_D = []
        for index in indices:
            conditional_dict = {}
            for i in range(0, set_length):
                if X_set[i][index] in conditional_dict.keys():
                    if y_set[i] in conditional_dict[X_set[i][index]].keys():
                        conditional_dict[X_set[i][index]][y_set[i]] += 1
                    else:
                        conditional_dict[X_set[i][index]][y_set[i]] = 1
                else:
                    conditional_dict[X_set[i][index]] = dict()
                    conditional_dict[X_set[i][index]][y_set[i]] = 1

            conditional_empire_entropy = 0
            feature_empire_entropy = 0
            for key in conditional_dict.keys():
                cond_dict = conditional_dict[key]
                conditional = sum(cond_dict.values()) / set_length
                conditional_empire_entropy += conditional * empire_entropy(cond_dict)
                feature_empire_entropy -= conditional * np.log2(conditional)
            HD_A.append(conditional_empire_entropy)
            H_A_D.append(feature_empire_entropy)

        g_HD_A = [HD - hd_a for hd_a in HD_A]

        g_r_HD_A = []
        for a, b in zip(g_HD_A, H_A_D):
            if b == 0:
                g_r_HD_A.append(a)
            else:
                g_r_HD_A.append(a / b)

        max_g_HD_A = max(g_HD_A)
        max_g_r_HD_A = max(g_r_HD_A)

        best_feature_index = -1
        if self.algorithm == "ID3":
            # Step 4: If max gain lower than epsilon, vote for the max class node
            if max_g_HD_A < self.epsilon:
                return TreeNode(type='leaf', belong=sorted(y_dict, key=lambda x: y_dict[x])[-1])

            # Step 5:
            best_feature_index = indices[g_HD_A.index(max_g_HD_A)]
        elif self.algorithm == "C4.5":
            if max_g_r_HD_A < self.epsilon:
                return TreeNode(type='leaf', belong=sorted(y_dict, key=lambda x: y_dict[x])[-1])

            best_feature_index = indices[g_r_HD_A.index(max_g_r_HD_A)]
        else:
            print("WTF of %s algorithm?", self.algorithm)

        # Build internal node using best_feature_index
        node = TreeNode(type='internal', index=best_feature_index, subtree={})

        new_subset = {}
        for i in range(0, set_length):
            if X_set[i][best_feature_index] in new_subset.keys():
                new_subset[X_set[i][best_feature_index]].append(i)
            else:
                new_subset[X_set[i][best_feature_index]] = [i]

        # Better to sort new_subset, subtree value range from value_min to value_max
        new_subset = dict((k, new_subset[k]) for k in sorted(new_subset.keys()))

        # Step 6:
        # indices.remove(best_feature_index)  # !!! FFFuck bug here
        sub_indices = list(filter(lambda x: x != best_feature_index, indices))

        for key in new_subset.keys():
            subset_list = new_subset[key]
            new_X_set = X_set[subset_list]
            new_y_set = y_set[subset_list]

            node.subtree[key] = self.build(new_X_set, new_y_set, sub_indices)

        return node

    @log
    def predict(self, X_test):
        n = len(X_test)
        d = X_test.shape[1]

        predict_label = np.full(n, -1)

        for i in range(0, n):
            to_predict = X_test[i]

            node = self.root
            while node.type != 'leaf':
                node = node.subtree[to_predict[node.index]]

            predict_label[i] = node.belong

            # print("Sample %d predicted as %d" % (i, predict_label[i]))

        return predict_label


def example_large():
    mnist_data = pd.read_csv("../data/mnist.csv")
    mnist_values = mnist_data.values

    images = mnist_values[::, 1::]
    labels = mnist_values[::, 0]

    X_train, X_test, y_train, y_test = train_test_split(
        images, labels, test_size=0.33, random_state=42
    )

    # Binary the image to avoid predict_probability gets 0
    binarizer_train = Binarizer(threshold=127).fit(X_train)
    X_train_binary = binarizer_train.transform(X_train)

    binarizer_test = Binarizer(threshold=127).fit(X_test)
    X_test_binary = binarizer_test.transform(X_test)

    # decision_tree = DecisionTree(algorithm="ID3", epsilon=0.01)
    decision_tree = DecisionTree(algorithm="C4.5", epsilon=0.01)

    print("Decision tree building...")
    decision_tree.train(X_train=X_train_binary, y_train=y_train)
    print("Building complete...")

    # Start predicting progress
    print("Testing on %d samples..." % len(X_test))
    y_predicted = decision_tree.predict(X_test=X_test_binary)

    calc_accuracy(y_pred=y_predicted, y_truth=y_test)


def example_small():
    X_train = np.asarray([[0, 0, 0, 0], [0, 0, 0, 1], [0, 1, 0, 1], [0, 1, 1, 0], [0, 0, 0, 0],
                          [1, 0, 0, 0], [1, 0, 0, 1], [1, 1, 1, 1], [1, 0, 1, 2], [1, 0, 1, 2],
                          [2, 0, 1, 2], [2, 0, 1, 1], [2, 1, 0, 1], [2, 1, 0, 2], [2, 0, 0, 0]])

    y_train = np.asarray([0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0])

    X_test = np.asarray([[0, 0, 0, 0], [0, 0, 0, 1], [0, 1, 0, 1], [0, 1, 1, 0], [0, 0, 0, 0],
                         [1, 0, 0, 0], [1, 0, 0, 1], [1, 1, 1, 1], [1, 0, 1, 2], [1, 0, 1, 2],
                         [2, 0, 1, 2], [2, 0, 1, 1], [2, 1, 0, 1], [2, 1, 0, 2], [2, 0, 0, 0]])

    y_test = np.asarray([0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0])

    decision_tree = DecisionTree(algorithm="ID3", epsilon=0.01)
    # decision_tree = DecisionTree(type="C4.5")

    print("Decision tree building...")
    decision_tree.train(X_train=X_train, y_train=y_train)
    print("Building complete...")

    # Start predicting progress
    print("Testing on %d samples..." % len(X_test))
    y_predicted = decision_tree.predict(X_test=X_test)

    calc_accuracy(y_pred=y_predicted, y_truth=y_test)


if __name__ == "__main__":
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # example_small()
    example_large()
