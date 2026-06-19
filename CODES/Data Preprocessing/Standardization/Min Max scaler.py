import pandas as pd
from sklearn.preprocessing import MinMaxScaler 
from sklearn.datasets import load_wine
import matplotlib.pyplot as plt

dataset = load_wine()
scaler = MinMaxScaler()

x = pd.DataFrame(data = dataset.data, columns=dataset.feature_names)
y = dataset.target
# print(x.head())

xnorm = scaler.fit_transform(x)
datax = pd.DataFrame(data=xnorm, columns=dataset.feature_names)
# print(datax.head())

plt.scatter(x["alcohol"],x["malic_acid"],c=y)
plt.scatter(datax["alcohol"],datax["malic_acid"],c=y)
plt.show()