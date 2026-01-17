import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler , PolynomialFeatures
from sklearn.linear_model import LinearRegression , SGDRegressor , Lasso
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

poly = PolynomialFeatures(degree=2)
scaler = StandardScaler()
sgd = SGDRegressor(max_iter=100000000,learning_rate="constant",random_state=None,eta0=0.01,penalty="l1")
lr = LinearRegression()
lasso  = Lasso(alpha=0.01,max_iter=100000000)
x , y = load_diabetes(return_X_y=True)

x = scaler.fit_transform(x)

trainx , testx , trainy , testy = train_test_split(x,y,test_size=0.3,random_state=33)

lr.fit(trainx,trainy)
lr_predy = lr.predict(testx)
print("The R2 score after Linear Regression is:",r2_score(testy,lr_predy))

sgd.fit(trainx,trainy)
sgd_predy = sgd.predict(testx)
print("The R2 score after Linear Regression Gradient Descent is:",r2_score(testy,sgd_predy))

lasso.fit(trainx,trainy)
lasso_predy = lasso.predict(testx)
print("The R2 score after Lasso Regression is:",r2_score(testy,lasso_predy))

poly_x = poly.fit_transform(x)
poly_x = scaler.fit_transform(poly_x)
poly_trainx , poly_testx , poly_trainy , poly_testy = train_test_split(poly_x,y,test_size=0.3,random_state=33)

lr.fit(poly_trainx,poly_trainy)
poly_lr_predy = lr.predict(poly_testx)
print("The R2 score after Polynomial Regression is:",r2_score(poly_testy,poly_lr_predy))

sgd.fit(poly_trainx,poly_trainy)
poly_sgd_predy = sgd.predict(poly_testx)
print("The R2 score after Polynomial Regression with Gradient Descent is:",r2_score(poly_testy,poly_sgd_predy))

lasso.fit(poly_trainx,poly_trainy)
poly_lasso_predy = lasso.predict(poly_testx)
print("The R2 score after Polynomial Lasso Regression is:",r2_score(poly_testy,poly_lasso_predy))