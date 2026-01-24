import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

x,y = load_diabetes(return_X_y=True, as_frame=True)
x = scaler.fit_transform(x)

trainx , testx ,trainy ,testy = train_test_split(x,y, test_size=0.3, random_state=33)

k = int(np.sqrt(trainx.shape[0]))
if k % 2 == 0:
    k += 1

knn = KNeighborsRegressor(n_neighbors=k)

knn.fit(trainx,trainy)
knn_predy = knn.predict(testx)
print("The R2 score after using Knn is:",r2_score(testy,knn_predy))
