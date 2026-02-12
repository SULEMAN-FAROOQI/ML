import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.datasets import fetch_covtype

data = fetch_covtype(as_frame=True)

x = data.data
y = data.target

iforest = IsolationForest(contamination=0.3)
forest_labels = iforest.fit_predict(x) 

# -1 for anomaly, 1 for normal

plt.scatter(x.iloc[:,0], x.iloc[:,1], c = forest_labels, cmap="viridis")
plt.show()

# pip install eif remember search when it can support python 3.13