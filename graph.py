import numpy as np
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import pairwise_distances

class Graph:
    def __init__(self, n_classes):
        self.n_classes = n_classes

    def fit(self, X, n_nodes):
        centers = KMeans(n_clusters=n_nodes).fit(X).cluster_centers_

        dist = pairwise_distances(centers)
        np.fill_diagonal(dist, np.inf)

        affinity = np.exp(dist / -np.std(dist.min(axis=1)))
        
        labels = SpectralClustering(n_clusters=self.n_classes, affinity='precomputed').fit_predict(affinity)

        self.clusters = [centers[labels == i] for i in range(self.n_classes)]

    def predict(self, x):
        diffs = [x - cluster[:, np.newaxis] for cluster in self.clusters]
        dists = [np.einsum('ijk,ijk->ij', df, df) for df in diffs]

        min_dists = np.array([np.min(dt, axis=0) for dt in dists])
        return np.argmin(min_dists, axis=0)
