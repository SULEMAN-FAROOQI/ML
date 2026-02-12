import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

x = np.random.randn(100, 2)

iforest = IsolationForest(contamination=0.1)
forest_labels = iforest.fit_predict(x) 

# -1 for anomaly, 1 for normal

plt.scatter(x[:,0], x[:,1], c = forest_labels, cmap="viridis")
plt.show()

# pip install eif remember search when it can support python 3.13