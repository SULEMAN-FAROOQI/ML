import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, make_moons
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

# Always scale before SVM

'''

Sicikit Learn provides the following classes for Support Vector Classification: 

1. SVC (C-Support Vector Classification): The most general and commonly used SVM classifier. It handles both 
linear and non-linear data through various kernel functions.

NuSVC (Nu-Support Vector Classification): Similar to SVC but uses a different parameter (nu) to control the 
number of support vectors and training errors.

LinearSVC (Linear Support Vector Classification): A faster implementation optimized specifically for 
linear kernels. It uses the liblinear library and is more efficient for large datasets where the number 
of samples is significantly larger than the number of features. 

Types of Kernels:

1. Linear: A simple linear kernel for linearly separable data.
2. Poly: A polynomial kernel, useful for data with polynomial relationship.
3. precomputed: Used when you manually compute a kernel (Gram) matrix, such as in graph kernels, string kernels, 
                or domain-specific similarity functions. 
4. rbf: The default kernel, widely used for non-linear problems.
5. sigmoid: A kernel function that resembles a neural network's activation function.

'''

svc1 = SVC(kernel="linear") 
svc2 = SVC(kernel="poly",degree=3,gamma=0.1) 
svc3 = SVC(kernel="rbf") 
svc4 = SVC(kernel="sigmoid") # Unstable 
# svc5 = SVC(kernel="precomputed") 

# Apply GridSearchCv for Hyper parameter tuning.

scaler = StandardScaler()

x0 , y0 = load_iris(return_X_y=True, as_frame=True)
x1, y1 = make_moons(n_samples=100, noise=0.15, random_state=42)

x0 = scaler.fit_transform(x0)
x1 = scaler.fit_transform(x1)

trainx0 , testx0 , trainy0 , testy0 = train_test_split(x0,y0, test_size=0.3, random_state=33)
trainx1 , testx1 , trainy1 , testy1 = train_test_split(x1,y1,test_size=0.3,random_state=33)

svc1.fit(trainx0,trainy0)
predy1_iris = svc1.predict(testx0)
print("The Accuracy score after taking linear kernel for linear dataset is:",accuracy_score(testy0,predy1_iris))

svc2.fit(trainx0,trainy0)
predy2_iris = svc2.predict(testx0)
print("The Accuracy score after taking polynomial kernel for linear dataset is:",accuracy_score(testy0,predy2_iris))

svc3.fit(trainx0,trainy0)
predy3_iris = svc3.predict(testx0)
print("The Accuracy score after taking rbf kernel for linear dataset is:",accuracy_score(testy0,predy3_iris))

svc4.fit(trainx0,trainy0) 
predy4_iris = svc4.predict(testx0)
print("The Accuracy score after taking sigmoid kernel for linear dataset is:",accuracy_score(testy0,predy4_iris))

# For non-linear dataset, linear kernel UNDERFITS.

svc2.fit(trainx1,trainy1)
predy1_moons = svc2.predict(testx1)
print("The Accuracy Score after taking polynomial Kernel for Non Linear dataset is:",accuracy_score(testy1,predy1_moons))

svc3.fit(trainx1,trainy1)
predy2_moons = svc3.predict(testx1)
print("The Accuracy Score after taking rbf Kernel for Non Linear dataset is:",accuracy_score(testy1,predy2_moons))

svc4.fit(trainx1,trainy1)
predy3_moons = svc4.predict(testx1)
print("The Accuracy Score after taking sigmoid Kernel for Non Linear dataset is:",accuracy_score(testy1,predy3_moons))

# Sigmoid Kernel usually behaves poorly.