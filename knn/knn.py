import numpy as np
from collections import Counter
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2)**2))


class KNN:
    def __init__(self, K = 3):
        self.K = K

    def fit(self, X, Y):
        self.X_train = X
        self.Y_train = Y

    def predict(self, X):
        predicted = [self._predict(x) for x in X]
        return predicted

    def _predict(self, x):
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        sorted_distance_indexes = np.argsort(distances)[:self.K]
        k_neighbor_labels = [self.Y_train[i] for i in sorted_distance_indexes]
        most_common = Counter(k_neighbor_labels).most_common(1)
        return most_common[0][0]



