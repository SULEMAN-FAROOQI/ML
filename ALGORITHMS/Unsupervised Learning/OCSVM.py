import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

scaler = StandardScaler()

data = pd.read_csv("Datasets\\anamoly.csv")

x = data.drop("fraud", axis=1)
y = data["fraud"]

x = scaler.fit_transform(x)

trainx ,testx, trainy ,testy = train_test_split(x,y, test_size=0.3, random_state=33)

ocs1 = OneClassSVM(kernel="linear", nu=0.5)
ocs2 = OneClassSVM(kernel="poly", nu=0.5)
# ocs3 = OneClassSVM(kernel="precomputed", nu=0.5) use precomputed when you actually compute the kernel matrix yourself.
ocs4 = OneClassSVM(kernel="rbf", nu=0.5)
ocs5 = OneClassSVM(kernel="sigmoid", nu=0.5)

ocs1.fit(trainx) # Unsupervised trainy will not be added
predy_ocs1 = ocs1.predict(testx)
predy_ocs1 = np.where(predy_ocs1 == -1, 1, 0)
print("The Accuracy Score after using OCSVM with linear kernel is:",accuracy_score(testy,predy_ocs1))

ocs2.fit(trainx) # Unsupervised trainy will not be added
predy_ocs2 = ocs2.predict(testx)
predy_ocs2 = np.where(predy_ocs2 == -1, 1, 0)
print("The Accuracy Score after using OCSVM with poly kernel is:",accuracy_score(testy,predy_ocs2))

'''

ocs3.fit(trainx) # Unsupervised trainy will not be added
predy_ocs3 = ocs3.predict(testx)
predy_ocs3 = np.where(predy_ocs3 == -1, 1, 0)
print("The Accuracy Score after using OCSVM with precomputed kernel is:",accuracy_score(testy,predy_ocs3))

'''

ocs4.fit(trainx) # Unsupervised trainy will not be added
predy_ocs4 = ocs4.predict(testx)
predy_ocs4 = np.where(predy_ocs4 == -1, 1, 0)
print("The Accuracy Score after using OCSVM with rbf kernel is:",accuracy_score(testy,predy_ocs4))

ocs5.fit(trainx) # Unsupervised trainy will not be added
predy_ocs5 = ocs5.predict(testx)
predy_ocs5 = np.where(predy_ocs5 == -1, 1, 0)
print("The Accuracy Score after using OCSVM with sigmoid kernel is:",accuracy_score(testy,predy_ocs5))

# Correct mapping: -1 → 1 (fraud), 1 → 0 (normal)