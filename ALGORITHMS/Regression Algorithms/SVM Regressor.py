import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes, make_friedman1
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

# Always scale before SVM

'''

Sicikit Learn provides the following classes for Support Vector Regression: 

1. SVR (Epsilon-Support Vector Regression): The standard regression model based on libsvm. It aims to fit the data 
                                            within a specified margin of tolerance epsilon.
2. NuSVR (Nu-Support Vector Regression): Similar to SVR but uses the Nu parameter to control the number of support vectors.

3. LinearSVR (Linear Support Vector Regression): A high-speed implementation for linear regression based on liblinear.

Types of Kernels:

1. Linear: A simple linear kernel for linearly separable data.
2. Poly: A polynomial kernel, useful for data with polynomial relationship.
3. precomputed: Used when you manually compute a kernel (Gram) matrix, such as in graph kernels, string kernels, 
                or domain-specific similarity functions. 
4. rbf: The default kernel, widely used for non-linear problems.
5. sigmoid: A kernel function that resembles a neural network's activation function.

'''

svc1 = SVR(kernel="linear") 
svc2 = SVR(kernel="poly",degree=3,gamma=0.1) 
svc3 = SVR(kernel="rbf") 
svc4 = SVR(kernel="sigmoid") # Unstable 
# svc5 = SVC(kernel="precomputed") 

# Apply GridSearchCv for Hyper parameter tuning.

scaler = StandardScaler()

x0 , y0 = load_diabetes(return_X_y=True, as_frame=True)
x1, y1 = make_friedman1(n_samples=500, n_features=10, noise=1.0, random_state=42)

x0 = scaler.fit_transform(x0)
x1 = scaler.fit_transform(x1)

trainx0 , testx0 , trainy0 , testy0 = train_test_split(x0,y0, test_size=0.3, random_state=33)
trainx1 , testx1 , trainy1 , testy1 = train_test_split(x1,y1,test_size=0.3,random_state=33)

svc1.fit(trainx0,trainy0)
predy1_diabetes = svc1.predict(testx0)
print("The R2 score after taking linear kernel for linear dataset is:",r2_score(testy0,predy1_diabetes))

svc2.fit(trainx0,trainy0)
predy2_diabetes = svc2.predict(testx0)
print("The R2 score after taking polynomial kernel for linear dataset is:",r2_score(testy0,predy2_diabetes))

svc3.fit(trainx0,trainy0)
predy3_diabetes = svc3.predict(testx0)
print("The R2 score after taking rbf kernel for linear dataset is:",r2_score(testy0,predy3_diabetes))

svc4.fit(trainx0,trainy0) 
predy4_diabetes = svc4.predict(testx0)
print("The R2 score after taking sigmoid kernel for linear dataset is:",r2_score(testy0,predy4_diabetes))

# For non-linear dataset, linear kernel UNDERFITS.

svc2.fit(trainx1,trainy1)
predy1_friedman1 = svc2.predict(testx1)
print("The R2 Score after taking polynomial Kernel for Non Linear dataset is:",r2_score(testy1,predy1_friedman1))

svc3.fit(trainx1,trainy1)
predy2_friedman1 = svc3.predict(testx1)
print("The R2 Score after taking rbf Kernel for Non Linear dataset is:",r2_score(testy1,predy2_friedman1))

svc4.fit(trainx1,trainy1)
predy3_friedman1 = svc4.predict(testx1)
print("The R2 Score after taking sigmoid Kernel for Non Linear dataset is:",r2_score(testy1,predy3_friedman1))

# Sigmoid Kernel usually behaves poorly.