import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as dendo

x, y = load_iris(return_X_y=True, as_frame=True)

dend = dendo.dendrogram(dendo.linkage(x,method="ward"))
plt.show()

ac = AgglomerativeClustering(n_clusters=3, linkage="ward")
labels = ac.fit_predict(x)

# Agglomerative clustering is unsupervised, so it does not use y

plt.scatter(x.iloc[:,0], x.iloc[:,1], c=labels, cmap="rainbow")
plt.show()

#It is a common approach to use the trained HAC model on the training data to generate labels, 
# then train a supervised classifier (like a K-Nearest Neighbors classifier) on that data to predict 
# the labels for new, incoming data.