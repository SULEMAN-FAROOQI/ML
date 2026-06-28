import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.datasets import load_digits

digits = load_digits()
x, y = digits.data, digits.target

# Embedding is the process of converting high dimensional into low dimesnions by tsne.

tsne = TSNE(n_components=2, perplexity=30, learning_rate='auto', init='pca', random_state=42) # n_components=2 for a 2D plot
x_embedded = tsne.fit_transform(x)

plt.figure(figsize=(10, 7))
scatter = plt.scatter(x_embedded[:, 0], x_embedded[:, 1], c=y, cmap='viridis', s=5)
plt.colorbar(scatter)
plt.title("t-SNE Visualization of Digits Dataset")
plt.show()