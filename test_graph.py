import numpy as np
import matplotlib.pyplot as plt
import umap
from datasets import load_dataset
import time
from sklearn.metrics import adjusted_rand_score
from graph import Graph

image_dataset = load_dataset('mnist', split='train', streaming=False, trust_remote_code=True)

np.random.seed(0)

X = np.stack([np.array(image).reshape(-1) for image in image_dataset['image']])
y = np.array(image_dataset['label'])

X_reduced = umap.UMAP(n_components=2).fit_transform(X)

plt.scatter(X_reduced[:, 0], X_reduced[:, 1], color='black', s=.5)

net = Graph(n_classes=10)

start = time.time()
net.fit(X_reduced, 150)
end = time.time()

colors = plt.cm.tab10(np.arange(10))

for target in np.unique(y):
    plt.scatter(X[(y == target), 0], X[(y == target), 1], label=f'Class {target}', s=.5)
    cluster = np.array(net.clusters[target])
    plt.scatter(cluster[:, 0], cluster[:, 1], linewidth=5, color=colors[target])

plt.legend()
plt.show()

print(f"Elapsed time: {end - start}")

guess = net.predict(X_reduced)

ari = adjusted_rand_score(y, guess)
print(f"Adjusted Rand Index: {ari}")
