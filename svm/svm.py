import numpy as np

class SVM:
    def __init__(self):
        super().__init__()
        self.w = None
        self.b = None


    def predict(self, x):
        linear_regression = np.dot(self.w, x) - self.b
        return np.sign(linear_regression)


    def fit(self, X, Y, epochs = 1, lr = 0.01, lp = 0.01, verbose = False):
        if len(X) != len(Y):
            raise TypeError("x and y should have same number of rows")
        Y = np.where(Y <=0, -1, 1)
        rows, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0

        for _  in range(epochs):
            for index, x in enumerate(X):
                condition = Y[index] * (np.dot( x, self.w) - self.b) >= 1
                if condition:
                    self.w -= lr * (2 * lp *  self.w)
                else:
                    self.w -= lr * (2 * lp * self.w - np.dot(x, Y[index]))
                    self.b -= lr * Y[index]





