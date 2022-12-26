# @Author: phd
# @Date: 2019-07-02
# @Site: github.com/phdsky
# @Description: NULL

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

    print("Predicting accuracy %f" % (hit_count / n))


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


class Node(object):
    def __init__(self, data, label, axis, left, right):
        self.data = data
        self.label = label
        self.axis = axis
        self.left = left
        self.right = right


class KDTree(object):
    def __init__(self, k, p):
        self.k = k
        self.p = p
        self.root = None

    def build(self, X_train, y_train):

        def create(dataset, axis):
            if len(dataset) == 0:
                return None  # Leaf node

            dataset.sort(key=lambda kv: kv[0][axis])  # Sort by axis

            median = len(dataset) // 2
            data = dataset[median][0]
            label = dataset[median][1]

            # This k is not self.k
            # Errata in book ?
            sp = (axis + 1) % len(data)

            left = create(dataset[0:median], sp)  # Create left sub-tree
            right = create(dataset[median + 1::], sp)  # Create right sub-tree

            return Node(data, label, axis, left, right)

        dataset = list(zip(X_train, y_train))
        self.root = create(dataset, 0)

    def nearest(self, x):
        nearest_nodes = []
        parent_nodes = []  # Parent nodes visited

        def traverse(x):
            while len(parent_nodes) != 0:
                parent_node = parent_nodes.pop()
                if parent_node is None:
                    continue

                dist = x[parent_node.axis] - parent_node.data[parent_node.axis]

                nearest_nodes.sort(key=lambda kv: kv[0])
                if abs(dist) < nearest_nodes[0][0]:
                    distance = minkowski(x, parent_node.data, self.p)
                    nearest_nodes.append((distance, parent_node))
                    parent_nodes.append(parent_node.right if dist < 0 else parent_node.left)

        # Find leaf node
        node = self.root
        while node is not None:
            parent_nodes.append(node)
            dist = x[node.axis] - node.data[node.axis]
            node = node.left if dist < 0 else node.right

        leaf_node = parent_nodes.pop()
        distance = minkowski(x, leaf_node.data, self.p)
        nearest_nodes.append((distance, leaf_node))

        traverse(x)

        nearest_nodes.sort(key=lambda kv: kv[0])
        print("Nearest neighbour is %s" % nearest_nodes[0][1].data)
        return nearest_nodes[0][1].label

    def predict(self, X_test):
        n = len(X_test)
        predict_label = np.full(n, -1)

        for i in range(0, n):
            predict_label[i] = self.nearest(X_test[i])
            print("Sample %d predicted as %d" % (i, predict_label[i]))

        return predict_label


def example_small():
    print("Start testing on simple dataset...")

    X_train = np.asarray([[2, 3], [5, 4], [9, 6], [4, 7], [8, 1], [7, 2]])
    y_train = np.asarray([0, 1, 2, 3, 4, 5])
    X_test = np.asarray([[3, 5]])
    # y_test = np.asarray([2])

    print("KDTree building...")
    kdtree = KDTree(k=1, p=2)  # Init KDTree
    kdtree.build(X_train=X_train, y_train=y_train)  # Build KDTree
    print("Building complete...")

    # Start predicting, training progress omitted
    print("Testing on %d samples..." % len(X_test))
    y_predicted = kdtree.predict(X_test=X_test)

    # calc_accuracy(y_pred=y_predicted, y_truth=y_test)

    print("Simple testing done...\n")


def example_large():
    mnist_data = pd.read_csv("../data/mnist.csv")
    mnist_values = mnist_data.values

    images = mnist_values[::, 1::]
    labels = mnist_values[::, 0]

    X_train, X_test, y_train, y_test = train_test_split(
        images, labels, test_size=100, random_state=42
    )

    print("KDTree building...")
    kdtree = KDTree(k=1, p=2)  # Init KDTree
    kdtree.build(X_train=X_train, y_train=y_train)  # Build KDTree
    print("Building complete...")

    # Start predicting, training progress omitted
    print("Testing on %d samples..." % len(X_test))
    y_predicted = kdtree.predict(X_test=X_test)

    calc_accuracy(y_pred=y_predicted, y_truth=y_test)


if __name__ == "__main__":
    example_small()
    # example_large()
