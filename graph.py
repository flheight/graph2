import numpy as np
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import pairwise_distances

class Graph:
    def __init__(self, n_classes):
        self.n_classes = n_classes

    def fit(self, X):
        max_iter = int(np.log2(X.shape[0]))
        kmeans = [KMeans(n_clusters=2**i).fit(X) for i in range(max_iter)]
        bic = [kmeans[i].inertia_ + .5 * (2**i + 1) * np.log(X.shape[0]) for i in range(max_iter)]

        best_kmeans = kmeans[np.minimum(np.argmin(bic) + 1, max_iter - 1)]
        centers = best_kmeans.cluster_centers_

        dist = pairwise_distances(centers, metric='sqeuclidean')
        np.fill_diagonal(dist, np.inf)

        affinity = np.exp(dist / -np.std(dist.min(axis=1)))
        
        labels = SpectralClustering(n_clusters=self.n_classes, affinity='precomputed').fit_predict(affinity)

        self.clusters = [centers[labels == i] for i in range(self.n_classes)]

    def predict(self, x):
        diffs = [x - cluster[:, np.newaxis] for cluster in self.clusters]
        dists = [np.einsum('ijk,ijk->ij', df, df) for df in diffs]

        min_dists = np.array([np.min(dt, axis=0) for dt in dists])
        return np.argmin(min_dists, axis=0)
