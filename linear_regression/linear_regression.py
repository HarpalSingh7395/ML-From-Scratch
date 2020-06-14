import numpy as np
import time
class LinearRegression:
    def __init__(self, learning_rate= 0.01):
        self.learning_rate = learning_rate
        self.weights = 0
        self.bias = 0


    def normalize(self, X):
        pass

    def predict(self, x):
        return np.dot(x, self.weights) + self.bias

    def mean_squared_error(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    def fit(self, x, y, epochs = 1, verbose = False):
        if len(x) != len(y):
            raise TypeError("x and y should have same number of rows")
        rows, features = x.shape
        self.weights = np.zeros(features)
        total_time = time.time()
        mse = []
        for _ in range(epochs):
            start_time = time.time()
            predicted = np.dot(x, self.weights) + self.bias
            w = (1/rows) * np.dot(x.T, (predicted - y))
            b = (1/rows) * np.sum(predicted - y)
            mse.append(self.mean_squared_error(y, predicted))
            if verbose:
                print("Epoch=" + str(_) + " time_taken=" + str((time.time() - start_time)* 1000) + "s mse=" + str(self.mean_squared_error(y, predicted)))
            self.weights -= self.learning_rate * w
            self.bias -= self.learning_rate * b
        if verbose:
            print("total time=" + str((time.time() - total_time)))
        print("Final MSE=" + str(self.mean_squared_error(y, predicted)))
        return mse




