import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor

x = np.random.randn(100, 2)

k = int(np.sqrt(x.shape[0]))
if k % 2 == 0:
    k += 1

lof = LocalOutlierFactor(n_neighbors=k, contamination=0.1)

# Returns -1 for outliers and 1 for inliers
lofy_label = lof.fit_predict(x)

plt.scatter(x[:,0], x[:,1], c=lofy_label, cmap="viridis")
plt.show()