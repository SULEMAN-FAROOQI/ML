from umap import UMAP
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt

digits = load_digits()
data = digits.data

reducer = UMAP(n_neighbors=5,
                    min_dist=0.1, # Sets how closely points can be packed togather in a cluster.
                    n_components=2) # Target Dimension)

embedding = reducer.fit_transform(data)

plt.figure(figsize=(10, 7))
plt.scatter(embedding[:, 0], embedding[:, 1], c=digits.target, cmap='Spectral', s=5)
plt.colorbar(boundaries=range(11))
plt.title('UMAP Projection of the Digits Dataset')
plt.show()

# It can be used for non-linear dimensionality reduction before clustering (like HDBSCAN).