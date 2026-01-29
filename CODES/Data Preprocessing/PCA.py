from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

df = pd.read_csv("Datasets\\MNIST.csv")
# print(df.sample(5))
# print(df.shape)

# plt.imshow(df.iloc[18306, 1:].values.reshape(28,28))
# plt.show()

x = df.drop(columns=["label"])
y = df["label"]

scaler = StandardScaler()

trainx,testx,trainy,testy = train_test_split(x,y,test_size=0.3,random_state=33)

trainx_s = scaler.fit_transform(trainx)
testx_s = scaler.fit_transform(testx)

for i in range(1,785):

    pca1 = PCA(n_components=i)
    trainx_pca = pca1.fit_transform(trainx_s)
    testx_pca = pca1.transform(testx_s)

    knn = KNeighborsClassifier()
    knn.fit(trainx,trainy)

    predy = knn.predict(testx)
    print("The Accuracy score at PC" ,i, "is" ,int(accuracy_score(testy,predy)))

'''

# For 2-Dimensional Visualization:

pca2d = PCA(n_components=2)
trainx2d = pca2d.fit_transform(trainx_s)

plt.scatter(trainx2d[:, 0], trainx2d[:, 1], c=trainy, cmap="tab10", s=6, alpha=0.9 )
plt.show()

'''

'''

# For 3_Dimensional Visualization:

pca3d = PCA(n_components=3)
trainx3d = pca3d.fit_transform(trainx_s)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(
    trainx3d[:, 0],  
    trainx3d[:, 1],  
    trainx3d[:, 2],  
    c=trainy,
    cmap="tab10", 
    s=6,               
    alpha=0.9          
)

ax.set_xlabel("PC 1")
ax.set_ylabel("PC 2")
ax.set_zlabel("PC 3")

plt.show()

'''

'''

# For Optimum Eigen Vectors and Values:

pca = PCA(n_components=None)
trx = pca.fit_transform(trainx_s)
tex = pca.transform(testx_s)

plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.show()

# This graph will tell us on which componenet our 90 percent variance is explained.

'''