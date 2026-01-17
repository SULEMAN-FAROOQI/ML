import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris

tree1 = DecisionTreeClassifier(criterion="entropy",splitter="best",max_depth=7)
tree2 = DecisionTreeClassifier(criterion="gini",splitter="best")
tree3 = DecisionTreeClassifier(criterion="entropy",splitter="random")
tree4 = DecisionTreeClassifier(criterion="gini",splitter="random")
tree5 = DecisionTreeClassifier(criterion="log_loss",splitter="best")
tree6 = DecisionTreeClassifier(criterion="log_loss",splitter="random")

x , y = load_iris(return_X_y=True, as_frame=True)

trainx , testx , trainy , testy = train_test_split(x,y,test_size=0.3,random_state=33)

tree1.fit(trainx,trainy)
predy1 = tree1.predict(testx)
print("The accuracy score after entropy and best splitting is:",accuracy_score(testy,predy1))

tree2.fit(trainx,trainy)
predy2 = tree2.predict(testx)
print("The accuracy score after gini and best splitting is:",accuracy_score(testy,predy2))

tree3.fit(trainx,trainy)
predy3 = tree3.predict(testx)
print("The accuracy score after entropy and random splitting is:",accuracy_score(testy,predy3))

tree4.fit(trainx,trainy)
predy4 = tree4.predict(testx)
print("The accuracy score after gini and random splitting is:",accuracy_score(testy,predy4))

tree5.fit(trainx,trainy)
predy5 = tree5.predict(testx)
print("The accuracy score after log loss and best splitting is:",accuracy_score(testy,predy5))

tree6.fit(trainx,trainy)
predy6 = tree6.predict(testx)
print("The accuracy score after log loss and random splitting is:",accuracy_score(testy,predy6))

