import numpy as np
import matplotlib.pyplot as plt
class KMeans:

    def __init__(self, K = 5, epochs = 100, verbose = False):
        self.K = K
        self.epochs = epochs
        self.verbose = verbose
        self.clusters = [[] for _ in range(self.K)]

        self.centroids = []

    def predict(self, X):
        self.X = X
        self.n_samples, self.n_features = X.shape

        random_sample_indexes = np.random.choice(self.n_samples, self.K, replace=False)
        self.centroids = [self.X[i] for i in random_sample_indexes]
        # print(random_sample_indexes)
        # print(self.centroids)
        for _ in range(self.epochs):
            self.clusters = self.makeClusters(self.centroids)
            break
            if self.verbose:
                self.plot()
            centroid_olds = self.centroids
            self.centroids = self.getNewCentroids(self.clusters)
            if self.isConverged(centroid_olds, self.centroids):
                break

        return self.getClusterLabels(self.clusters)


    def getClusterLabels(self, clusters):
        labels = np.empty(self.n_samples)
        for cluster_index, cluster in enumerate(clusters):
            for sample_index in cluster:
                labels[sample_index] = cluster_index
        return labels



    def isConverged(self, old_centroids, new_centroids):
        distances = [self.euclieanDistance(old_centroids[i], new_centroids[i]) for i in range(self.K)]
        return sum(distances) == 0

    def getNewCentroids(self, clusters):
        centroids = np.zeros((self.K, self.n_features))
        for index, cluster in enumerate(clusters):
            cluster_mean = np.mean(self.X[cluster], axis=0)
            centroids[index] = cluster_mean
        return centroids

    def makeClusters(self, centroids):
        clusters = [[] for _ in range(self.K)]
        for sample_index, sample in enumerate(self.X):
            centroid_index = self.closestCentroid(sample, centroids)
            clusters[centroid_index].append(sample_index)
        return clusters

    def closestCentroid(self, sample, centroids):
        distances = [self.euclieanDistance(sample,centroid) for centroid in centroids]
        closest_index = np.argmin(distances)
        return closest_index

    def plot(self):
        fig, ax = plt.subplots(figsize=(12, 8))

        for i, index in enumerate(self.clusters):
            point = self.X[index].T
            ax.scatter(*point)

        for point in self.centroids:
            ax.scatter(*point, marker="x", color='black', linewidth=2)

        plt.show()

    def euclieanDistance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2)**2))

