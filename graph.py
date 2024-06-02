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

        best_kmeans = kmeans[np.argmin(bic) + 1]
        centers = best_kmeans.cluster_centers_

        dist = 20*pairwise_distances(centers, metric='sqeuclidean')
        np.fill_diagonal(dist, np.inf)

        guess = best_kmeans.predict(X)
        
        for i in range(dist.shape[0]):
            data_i = X[guess == i]
            for j in range(i):
                points_ij = np.vstack((centers[i], centers[j]))
                data_j = X[guess == j]
                data = np.vstack((data_i, data_j))

                diffs = data - points_ij[:, np.newaxis]
                dists = np.min(np.einsum('ijk,ijk->ij', diffs, diffs), axis=0)
                intertia_points = dists.mean()
                
                segment = (centers[j] - centers[i]).reshape(1, -1)
                projs = np.dot(data - centers[i], segment.T) / np.einsum('ij,ij->i', segment, segment)
                nearests = centers[i] + np.outer(np.clip(projs, 0, 1), segment)
                diffs = data - nearests
                dists = np.einsum('ij,ij->i', diffs, diffs)
                intertia_segment = dists.mean()

                dist[i, j] += intertia_segment - intertia_points

        dist = dist + dist.T

        print(dist)

        affinity = np.exp(dist / -np.var(dist.min(axis=1)))
        
        labels = SpectralClustering(n_clusters=self.n_classes, affinity='precomputed').fit_predict(affinity)

        self.clusters = [centers[labels == i] for i in range(self.n_classes)]

    def predict(self, x):
        diffs = [x - cluster[:, np.newaxis] for cluster in self.clusters]
        dists = [np.einsum('ijk,ijk->ij', df, df) for df in diffs]

        min_dists = np.array([np.min(dt, axis=0) for dt in dists])
        return np.argmin(min_dists, axis=0)
