# Also Known as Multinomial Logistic Regression.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score , confusion_matrix , precision_score , recall_score , f1_score , classification_report
from sklearn.model_selection import train_test_split

# ovr in multiclass operates with applying the sigmoid fn of logistic regression on each class.
# multinomial in multiclass opearates with applying the softmax fn on each class

scaler  = StandardScaler()
log1 = LogisticRegression(solver="lbfgs",multi_class="multinomial") # cross entropy
log2 = LogisticRegression(solver="newton-cg",multi_class="multinomial") # cross entropy
log3 = LogisticRegression(solver="newton-cholesky",multi_class="multinomial") # cross entropy
log4 = LogisticRegression(solver="sag",multi_class="multinomial") # Closer to GD
log5 = LogisticRegression(solver="saga",multi_class="multinomial") # Closer to GD
log6 = LogisticRegression(solver="liblinear",multi_class="ovr") # Hybrid and does not work for multinomial

x,y = load_iris(return_X_y=True, as_frame=True)

plt.scatter(x["sepal length (cm)"],x["sepal width (cm)"], c = y)
plt.show()

x = scaler.fit_transform(x)
trainx , testx , trainy , testy = train_test_split(x,y,test_size=0.3,random_state=33)

log1.fit(trainx,trainy)
predy1 = log1.predict(testx)
print("The accuracy score after using lbfgs is:",accuracy_score(testy,predy1))
print("The confusion matrix after using lbfgs is:\n",confusion_matrix(testy,predy1))
print("The precision score after using lbfgs is:",precision_score(testy,predy1,average="macro"))
print("The recall score after using lbfgs is:",recall_score(testy,predy1,average="macro"))
print("The f1 score after using lbfgs is:",f1_score(testy,predy1,average="macro"))
print("The Classification report after using lbfgs is:\n",classification_report(testy,predy1))

log2.fit(trainx,trainy)
predy2 = log2.predict(testx)
print("The accuracy score after using newton-cg is:",accuracy_score(testy,predy2))
print("The confusion matrix after using newton-cg is:\n",confusion_matrix(testy,predy2))
print("The precision score after using newton-cg is:",precision_score(testy,predy2,average="macro"))
print("The recall score after using newton-cg is:",recall_score(testy,predy2,average="macro"))
print("The f1 score after using newton-cg is:",f1_score(testy,predy2,average="macro"))
print("The Classification report after using newton-cg is:\n",classification_report(testy,predy2))

log3.fit(trainx,trainy)
predy3 = log3.predict(testx)
print("The accuracy score after using newton-cholesky is:",accuracy_score(testy,predy3))
print("The confusion matrix after using newton-cholesky is:\n",confusion_matrix(testy,predy3))
print("The precision score after using newton-cholesky is:",precision_score(testy,predy3,average="macro"))
print("The recall score after using newton-cholesky is:",recall_score(testy,predy3,average="macro"))
print("The f1 score after using newton-cholesky is:",f1_score(testy,predy3,average="macro"))
print("The Classification report after using newton-cholesky is:\n",classification_report(testy,predy3))

log4.fit(trainx,trainy)
predy4 = log4.predict(testx)
print("The accuracy score after using sag is:",accuracy_score(testy,predy4))
print("The confusion matrix after using sag is:\n",confusion_matrix(testy,predy4))
print("The precision score after using sag is:",precision_score(testy,predy4,average="macro"))
print("The recall score after using sag is:",recall_score(testy,predy4,average="macro"))
print("The f1 score after using sag is:",f1_score(testy,predy4,average="macro"))
print("The Classification report after using sag is:\n",classification_report(testy,predy4))

log5.fit(trainx,trainy)
predy5 = log5.predict(testx)
print("The accuracy score after using saga is:",accuracy_score(testy,predy5))
print("The confusion matrix after using saga is:\n",confusion_matrix(testy,predy5))
print("The precision score after using saga is:",precision_score(testy,predy5,average="macro"))
print("The recall score after using saga is:",recall_score(testy,predy5,average="macro"))
print("The f1 score after using saga is:",f1_score(testy,predy5,average="macro"))
print("The Classification report after using saga is:\n",classification_report(testy,predy5))

log6.fit(trainx,trainy)
predy6 = log6.predict(testx)
print("The accuracy score after using liblinear is:",accuracy_score(testy,predy6))
print("The confusion matrix after using liblinear is:\n",confusion_matrix(testy,predy6))
print("The precision score after using liblinear is:",precision_score(testy,predy6,average="macro"))
print("The recall score after using liblinear is:",recall_score(testy,predy6,average="macro"))
print("The f1 score after using liblinear is:",f1_score(testy,predy6,average="macro"))
print("The Classification report after using liblinear is:\n",classification_report(testy,predy6))
