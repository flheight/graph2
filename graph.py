import numpy as np
from sklearn.cluster import KMeans, SpectralClustering
from scipy.spatial.distance import pdist, squareform
from scipy.stats import trim_mean

class Graph:
    def __init__(self, n_classes):
        self.n_classes = n_classes

    def fit(self, X, n_nodes):
        centers = KMeans(n_clusters=n_nodes).fit(X).cluster_centers_

        k_neigbors=n_nodes // self.n_classes

        dist = squareform(pdist(centers))
        np.fill_diagonal(dist, np.inf)
        sigma = np.sort(dist, axis=0)[:k_neigbors]
        sigma = np.std(sigma, axis=0) * np.power(.75 * k_neigbors, -.2)
        gamma = .5 / (sigma[np.newaxis, :] * sigma[:, np.newaxis])

        print(gamma.mean())

        affinity_matrix = np.exp(-dist**2 * gamma)

        labels = SpectralClustering(n_clusters=self.n_classes, affinity='precomputed').fit_predict(affinity_matrix)

        self.clusters = [centers[labels == i] for i in range(self.n_classes)]

    def predict(self, x):
        diffs = [x - cluster[:, np.newaxis] for cluster in self.clusters]
        dists = [np.einsum('ijk,ijk->ij', df, df) for df in diffs]

        min_dists = np.array([np.min(dt, axis=0) for dt in dists])
        return np.argmin(min_dists, axis=0)