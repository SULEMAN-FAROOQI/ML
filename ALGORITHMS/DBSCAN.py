from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import numpy as np

x,y = make_moons(n_samples=200, noise=0.3, random_state=42)

dbscan = DBSCAN(eps=0.3, min_samples=5)
labels = dbscan.fit_predict(x)

# Agglomerative clustering is unsupervised, so it does not use y

plt.scatter(x[:,0], x[:,1], c=labels, cmap="viridis")
plt.show()

#It is a common approach to use the trained HAC model on the training data to generate labels, 
# then train a supervised classifier (like a K-Nearest Neighbors classifier) on that data to predict 
# the labels for new, incoming data.