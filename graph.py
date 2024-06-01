import numpy as np
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import pairwise_distances

class Graph:
    def __init__(self, n_classes):
        self.n_classes = n_classes

    def fit(self, X, n_nodes):
        centers = KMeans(n_clusters=n_nodes).fit(X).cluster_centers_

        k_neighbors = n_nodes // self.n_classes

        dist = pairwise_distances(centers)
        np.fill_diagonal(dist, np.inf)
        sorted_dist = np.sort(dist, axis=0)[:k_neighbors]
        sigma = np.max(np.diff(sorted_dist, axis=0), axis=0).mean()

        affinity_matrix = np.exp(-dist**2  / sigma**2)

        labels = SpectralClustering(n_clusters=self.n_classes, affinity='precomputed').fit_predict(affinity_matrix)

        self.clusters = [centers[labels == i] for i in range(self.n_classes)]

    def predict(self, x):
        diffs = [x - cluster[:, np.newaxis] for cluster in self.clusters]
        dists = [np.einsum('ijk,ijk->ij', df, df) for df in diffs]

        min_dists = np.array([np.min(dt, axis=0) for dt in dists])
        return np.argmin(min_dists, axis=0)
