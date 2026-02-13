import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import make_friedman1

x , y = make_friedman1(n_samples=1000,noise=1.0,random_state=42)

tree1 = DecisionTreeRegressor(criterion="absolute_error",splitter="best")
tree2 = DecisionTreeRegressor(criterion="friedman_mse",splitter="best")
# tree3 = DecisionTreeRegressor(criterion="poisson",splitter="best") some values of y are negative
tree4 = DecisionTreeRegressor(criterion="squared_error",splitter="best")
tree5 = DecisionTreeRegressor(criterion="absolute_error",splitter="random")
tree6 = DecisionTreeRegressor(criterion="friedman_mse",splitter="random")
# tree7 = DecisionTreeRegressor(criterion="poisson",splitter="random") some values of y are negative
tree8 = DecisionTreeRegressor(criterion="squared_error",splitter="random")

trainx , testx , trainy , testy = train_test_split(x,y,test_size=0.3,random_state=33)

tree1.fit(trainx,trainy)
predy1 = tree1.predict(testx)
print("The R2 score after using absolute error and best splitting is:",r2_score(testy,predy1))

tree2.fit(trainx,trainy)
predy2 = tree2.predict(testx)
print("The R2 score after using friedman mse and best splitting is:",r2_score(testy,predy2))

tree4.fit(trainx,trainy)
predy4 = tree4.predict(testx)
print("The R2 score after using squared error and best splitting is:",r2_score(testy,predy4))

tree5.fit(trainx,trainy)
predy5 = tree5.predict(testx)
print("The R2 score after using absolute error and random splitting is:",r2_score(testy,predy5))

tree6.fit(trainx,trainy)
predy6 = tree6.predict(testx)
print("The R2 score after using friedman mse and random splitting is:",r2_score(testy,predy6))

tree8.fit(trainx,trainy)
predy8 = tree8.predict(testx)
print("The R2 score after using squared error and random splitting is:",r2_score(testy,predy8))
