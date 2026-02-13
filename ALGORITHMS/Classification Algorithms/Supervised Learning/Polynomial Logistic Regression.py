import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures , StandardScaler
from sklearn.datasets import make_circles
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report , confusion_matrix
from sklearn.model_selection import train_test_split

scaler = StandardScaler()
poly = PolynomialFeatures(degree=3)
log1 = LogisticRegression(solver="lbfgs") # cross entropy
log2 = LogisticRegression(solver="newton-cg") # cross entropy
log3 = LogisticRegression(solver="newton-cholesky") # cross entropy
log4 = LogisticRegression(solver="sag") # Closer to GD
log5 = LogisticRegression(solver="saga") # Closer to GD
log6 = LogisticRegression(solver="liblinear") # Hybrid

x,y = make_circles(n_samples=1000,noise=0.1, factor=0.5)

plt.scatter(x[:,0],x[:,1],c=y)
plt.show()

x = poly.fit_transform(x)
x = scaler.fit_transform(x)

trainx , testx , trainy , testy = train_test_split(x,y,test_size=0.3,random_state=33)

log1.fit(trainx,trainy)
predy1 = log1.predict(testx)
print("The confusion matrix after using lbfgs is:\n",confusion_matrix(testy,predy1))
print("The Classification report after using lbfgs is:\n",classification_report(testy,predy1))

log2.fit(trainx,trainy)
predy2 = log2.predict(testx)
print("The confusion matrix after using newton-cg is:\n",confusion_matrix(testy,predy2))
print("The Classification report after using newton-cg is:\n",classification_report(testy,predy2))

log3.fit(trainx,trainy)
predy3 = log3.predict(testx)
print("The confusion matrix after using newton-cholesky is:\n",confusion_matrix(testy,predy3))
print("The Classification report after using newton-cholesky is:\n",classification_report(testy,predy3))

log4.fit(trainx,trainy)
predy4 = log4.predict(testx)
print("The confusion matrix after using sag is:\n",confusion_matrix(testy,predy4))
print("The Classification report after using sag is:\n",classification_report(testy,predy4))

log5.fit(trainx,trainy)
predy5 = log5.predict(testx)
print("The confusion matrix after using saga is:\n",confusion_matrix(testy,predy5))
print("The Classification report after using saga is:\n",classification_report(testy,predy5))

log6.fit(trainx,trainy)
predy6 = log6.predict(testx)
print("The confusion matrix after using liblinear is:\n",confusion_matrix(testy,predy6))
print("The Classification report after using liblinear is:\n",classification_report(testy,predy6))
