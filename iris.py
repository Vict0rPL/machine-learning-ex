import numpy as np
from sklearn import datasets
from sklearn.cluster import KMeans
from itertools import cycle, combinations
import matplotlib.pyplot as pl

iris = datasets.load_iris()

km = KMeans(n_clusters=3)
km.fit(iris.data)

predictions = km.predict(iris.data)

colors = cycle('rgb')
labels = ["Claster 1", "Claster 2", "claster 3"]
targets = range(len(labels))

feature_index = range(len(iris.feature_names))
feature_names = iris.feature_names
combs = combinations(feature_index, 2)
f, axarr = pl.subplots(3, 2)
ax_flat = axarr.flat

for comb, axflat in zip(combs, ax_flat):
    for target, color, label in zip(targets, colors, labels):
        feature_index_x = comb[0]
        feature_index_y = comb[1]

        axflat.scatter(iris.data[predictions == target, feature_index_x], iris.data[predictions == target, feature_index_y], c=color, label=label)
        axflat.set_xlabel(feature_names[feature_index_x])
        axflat.set_ylabel(feature_names[feature_index_y])

f.tight_layout()
pl.show()

