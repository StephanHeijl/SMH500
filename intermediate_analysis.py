from sklearn.manifold import *
from sklearn.cluster import *
import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(linewidth=200, suppress=True)

data = np.load("reduced.npy")
classes = np.argmax(np.load("classes.npy"), axis=-1)
preds = np.argmax(np.load("preds.npy"), axis=-1)
timeseries = np.load("timeseries.npy")

#man = TSNE(n_components=2, init="pca", random_state=0)
man = Isomap(n_components=2)
X = man.fit_transform(data)

cluster = AffinityPropagation()
clustered = cluster.fit_predict(X)

plt.scatter(X[:, 0], X[:, 1], c=clustered, marker="o")
plt.show()

plt.scatter(X[:, 0], X[:, 1], c=preds, marker="o")
plt.show()

timeseries = timeseries.reshape(timeseries.shape[:2])

for cluster in np.unique(clustered):
    idx = (clustered == cluster)
    for series in timeseries[idx]:
        plt.plot(series, c="silver", linewidth=1)

    avg = timeseries[idx].mean(axis=0)
    plt.plot(avg, c="red", linewidth=1)
    plt.title("Cluster %i, average class: %.2f, average pred: %.2f" % (
        cluster, classes[idx].mean(), preds[idx].mean()
    ))
    plt.show()

exit()

markers = list("voxs^")
for c in classes:
    plt.scatter(X[classes==c, 0], X[classes==c, 1], c=preds[classes==c], marker=markers[c])

plt.colorbar()
plt.show()
