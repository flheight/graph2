import numpy as np
import matplotlib.pyplot as plt
import umap
from sklearn.datasets import make_moons
from graph import Graph

np.random.seed(0)

X, y = make_moons(n_samples=1024, noise=0.1, random_state=42)

plt.scatter(X[:, 0], X[:, 1], color='black', s=.5)

net = Graph(n_classes=2)

net.fit(X, 15)

colors = plt.cm.tab10(np.arange(2))

guess = net.predict(X)

for i in range(2):
    plt.scatter(net.clusters[i][:, 0], net.clusters[i][:, 1], color=colors[i], label=f'Cluster {i}')

plt.show()