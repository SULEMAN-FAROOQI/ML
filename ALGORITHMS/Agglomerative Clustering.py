import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as dendo

X, y = load_iris(return_X_y=True, as_frame=True)
X = X[["sepal length (cm)", "sepal width (cm)"]]

dend = dendo.dendrogram(dendo.linkage(X,method="ward"))
plt.show()

c1 = AgglomerativeClustering(n_clusters=3, linkage="ward")
labels = c1.fit_predict(X)

plt.scatter(X.iloc[:,0], X.iloc[:,1], c=labels, cmap="rainbow")
plt.show()

#It is a common approach to use the trained HAC model on the training data to generate labels, 
# then train a supervised classifier (like a K-Nearest Neighbors classifier) on that data to predict 
# the labels for new, incoming data