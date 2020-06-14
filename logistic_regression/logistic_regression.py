import numpy as np
import time 
class LogisticRegression:
    def __init__(self ):
        super().__init__()
        self.bias = None
        self.weights = None
      
    
    def predict(self, X):
        linear_hypothesis = np.dot(X, self.weights) + self.bias
        y_predicted = self.sigmoid(linear_hypothesis)
        return [1 if i > 0.5 else 0 for i in y_predicted]
    
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def accuracy(self, y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy
    
    def fit(self, X, Y, epochs = 1, lr = 0.01, verbose = False):
        if len(X) != len(Y):
            raise TypeError("x and y should have same number of rows")
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        total_time = time.time()
        accuracy_array = []
        for _ in range(epochs):
            start_time = time.time()
            linear_hypothesis = np.dot(X, self.weights) + self.bias
            y_perdicted = self.sigmoid(linear_hypothesis)
            accuracy = self.accuracy(Y,y_perdicted)
            accuracy_array.append(accuracy)
            if verbose:
                print("Epoch=" + str(_) + " time_taken=" + str((time.time() - start_time)* 1000) + "s accuracy=" + str(accuracy))
            dw = (1 / n_samples) * np.dot(2 * X.T, (y_perdicted - Y))
            db = (1 / n_samples) * np.sum(2 * (y_perdicted - Y))
            
            self.weights -= lr * dw
            self.bias -= lr * db
        if verbose:
            print("total time=" + str((time.time() - total_time)))
        print("Final Accuracy=" + str(self.accuracy(Y, y_perdicted)))
        return accuracy_array
