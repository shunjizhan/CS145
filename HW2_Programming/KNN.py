from sklearn import cross_validation
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import math


def load_dataset():
    x = np.zeros((150, 4))
    y = np.zeros(150, dtype=int)
    instance_index = 0
    with open("iris.data", 'r') as f:
        for line in f:
            data = line.strip().split(',')
            x[instance_index] = data[0:4]
            if data[4] == 'Iris-setosa':
                y[instance_index] = 0
            elif data[4] == 'Iris-versicolor':
                y[instance_index] = 1
            else:
                y[instance_index] = 2
            instance_index += 1

    perm = np.random.permutation(np.arange(x.shape[0]))
    x = x[perm]
    y = y[perm]
    return x, y


def euclidean_distance(x1, x2):
    assert(len(x1) == len(x2))

    total = 0
    for i in range(len(x1)):
        total += (x1[i] - x2[i]) ** 2

    return math.sqrt(total)


# The function finds the class labels of K nearest neighbours for the given test data point 'j'
# Uses function euclidean_distance
# train_x and train_y are the training data points and their class labels
# test_x_j is the 'jth' test point whose neighbours need to be found
# Returns a 1-d numpy array with class labels of nearest neighbours
def find_K_neighbours(train_x, train_y, test_x_j, K):
    assert(len(train_y) >= 5)

    neighbours_y = []
    for i in range(len(train_x)):
        y = train_y[i]
        x = train_x[i]
        distance = euclidean_distance(x, test_x_j)
        neighbours_y.append((distance, y))

    all_neighbors = sorted(neighbours_y)
    nearest_neighbors = all_neighbors[:K]

    return [p[1] for p in nearest_neighbors]


# The function classifies a data point given the labels of its nearest neighbours
# Returns the label for the data point
def classify(neighbours_y):
    label_count = Counter(neighbours_y)
    return label_count.most_common(1)[0][0]


if __name__ == '__main__':
    x, y = load_dataset()
    cv = cross_validation.KFold(len(x), n_folds=5)

    average_accuracies = np.zeros(119)

    for K in range(1, 120):
        fold_accuracies = []
        for traincv, testcv in cv:
            train_x = x[traincv]
            train_y = y[traincv]
            test_x = x[testcv]
            test_y = y[testcv]

            predicted_labels = np.full(test_x.shape[0], -1, dtype=int)

            for j in range(test_x.shape[0]):
                neighbours_y = find_K_neighbours(train_x, train_y, test_x[j], K)
                predicted_labels[j] = classify(neighbours_y)

            fold_accuracies.append(np.mean(predicted_labels == test_y))

        average_accuracies[K-1] = np.mean(fold_accuracies)

    print("Average accuracies with 5-fold cross validation for K varying from 1 to 119:")
    print(average_accuracies)

    print("Best value of K: ")
    print(np.argmax(average_accuracies) + 1)

    x = np.arange(1, 120)
    plt.plot(x, average_accuracies)
    plt.show()





