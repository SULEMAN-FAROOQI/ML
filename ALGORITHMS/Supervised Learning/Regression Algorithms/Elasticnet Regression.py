import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures , StandardScaler
from sklearn.linear_model import LinearRegression , SGDRegressor , ElasticNet
from sklearn.metrics import r2_score
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

scaler =  StandardScaler()
poly = PolynomialFeatures(degree=2)
lr  = LinearRegression()
sgd = SGDRegressor(max_iter=100000000, penalty='elasticnet', eta0=0.5, learning_rate='constant', random_state=None)
elastic  = ElasticNet(alpha=0.25, l1_ratio=0.3, max_iter=100000000)
x , y = load_diabetes(return_X_y=True)

x = scaler.fit_transform(x)
trainx , testx , trainy, testy = train_test_split(x,y,test_size=0.3,random_state=33)

lr.fit(trainx,trainy)
lr_predy = lr.predict(testx)
print("The R2 score after Linear Regression is:",r2_score(testy,lr_predy))

sgd.fit(trainx,trainy)
sgd_predy = sgd.predict(testx)
print("The R2 score after Linear Regression with gradient descent is:",r2_score(testy,sgd_predy))

elastic.fit(trainx,trainy)
elastic_predy = elastic.predict(testx)
print("The R2 score after Elastinet Regression is:",r2_score(testy,elastic_predy))

poly_x = poly.fit_transform(x)
poly_x = scaler.fit_transform(poly_x)

poly_trainx , poly_testx , poly_trainy , poly_testy = train_test_split(poly_x,y,test_size=0.3,random_state=33)

lr.fit(poly_trainx,poly_trainy)
poly_lr_predy = lr.predict(poly_testx)
print("The R2 score after Polynomial Linear regression is:",r2_score(poly_testy,poly_lr_predy))

sgd.fit(poly_trainx,poly_trainy)
poly_sgd_predy = sgd.predict(poly_testx)
print("The Polynomial Linear Regression with Gradient Descent is:",r2_score(poly_testy,poly_sgd_predy))

elastic.fit(poly_trainx,poly_trainy)
poly_elastic_predy = elastic.predict(poly_testx)
print("The R2 score after Polynomial Elasticnet Regression is:",r2_score(poly_testy,poly_elastic_predy))
