import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

x , y = load_iris(return_X_y=True, as_frame=True)

print(y)

# Elbow Method:

'''

inertia = []
for i in range(1,19):
    km = KMeans(n_clusters=i)
    km.fit_predict(x) 
    inertia.append(km.inertia_)

print(inertia)

plt.plot(range(1,19),inertia) 
plt.show()

'''

# By Graph of elbow method the best value of k is 3.
# fit_predict trains the model and also assign a value to the clusters accordingly.

trainx, testx, trainy , testy = train_test_split(x,y, test_size=0.3, random_state=33)

km = KMeans(n_clusters=3)
km.fit(trainx,trainy)
predy = km.predict(testx)
print("The Accuracy score after K means clustering is:",accuracy_score(testy,predy))

