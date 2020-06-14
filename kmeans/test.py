import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from kmeans import KMeans

# X, y = make_blobs(centers=4, n_samples=500, n_features=2, shuffle=True, random_state=42)
X, y = make_blobs(centers=3, n_samples=500, n_features=2, shuffle=True, random_state=40)
print(X.shape)

clusters = len(np.unique(y))
# print(clusters)
k = KMeans(K = clusters, epochs = 100, verbose = False)
labels = k.predict(X)
k.plot()
# print(labels)