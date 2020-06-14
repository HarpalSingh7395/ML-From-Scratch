import  numpy as np


class PCA:

    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None


    def fit(self, X):
        n_samples, n_features = X.shape
        # Finding mean
        self.mean =  np.sum(np.matmul(X, X.T))/n_samples
        X = X - self.mean
        # Covariance matrix
        cov = np.cov(X.T)
        # eigen vectors, eigen values
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        # sort eigen vectors
        eigenvectors = eigenvectors.T
        indexes = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[indexes]
        eigenvectors = eigenvectors[indexes]
        self.components = eigenvectors[0:self.n_components]



    def transform(self, X):
        X = X - self.mean
        return np.dot(X, self.components.T)