import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

scaler = StandardScaler()

data = pd.read_csv("Datasets\\anamoly.csv")

x = data.drop("fraud", axis=1)
y = data["fraud"]

x = scaler.fit_transform(x)

trainx ,testx, trainy ,testy = train_test_split(x,y, test_size=0.3, random_state=33)

k = int(np.sqrt(trainx.shape[0]))
if k % 2 == 0:
    k += 1

lof = LocalOutlierFactor(n_neighbors=k, novelty=True) # novelity enables prediction on new data

lof.fit(trainx) # Unsupervised trainy will not be added
predy_lof = lof.predict(testx)

# Correct mapping: -1 → 1 (fraud), 1 → 0 (normal)

predy_lof = np.where(predy_lof == -1, 1, 0)

print("The Accuracy score after using local factor outlier for anamoly detection is:",accuracy_score(testy,predy_lof))