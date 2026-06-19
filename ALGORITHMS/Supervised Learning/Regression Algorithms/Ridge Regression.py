'''

# Ridge regresssion with OLS:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.linear_model import Ridge, LinearRegression , BayesianRidge
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures , StandardScaler

# alpha is lambda where the value of alpha is between 0 and infinity.

lr = LinearRegression()
ridge1 = Ridge(alpha=0.01, solver="svd") 
ridge2 = Ridge(alpha=0.01,solver='cholesky')
ridge3 = BayesianRidge(alpha_1=0.01, alpha_2=0.001, lambda_1=0.001, lambda_2=0.0001) # Uses a probabilistic approach
poly = PolynomialFeatures(degree=2)
scaler = StandardScaler()

x,y = load_diabetes(return_X_y=True)
simple_x = scaler.fit_transform(x)

trainx , testx, trainy, testy = train_test_split(simple_x,y,test_size=0.3,random_state=33)

lr.fit(trainx,trainy)
lr_predy = lr.predict(testx)
print("The r2 score after linear regression is:",r2_score(testy,lr_predy))

ridge1.fit(trainx,trainy)
ridge1_predy = ridge1.predict(testx)
print("The r2 score after ridge regression in svd is:",r2_score(testy,ridge1_predy))

ridge2.fit(trainx,trainy)
ridge2_predy = ridge2.predict(testx)
print("The r2 score after ridge regression in cholesky is:",r2_score(testy,ridge2_predy))

ridge3.fit(trainx,trainy)
ridge3_predy = ridge3.predict(testx)
print("The r2 score after Bayesian ridge regression is:",r2_score(testy,ridge3_predy))

poly_x = poly.fit_transform(x)
poly_x = scaler.fit_transform(poly_x)

# Polynomial features must be scaled otherwise Large feature magnitudes dominate and Regularization behaves incorrectly.

poly_trainx , poly_testx , trainy ,testy = train_test_split(poly_x,y,test_size=0.3,random_state=33)

ridge1.fit(poly_trainx,trainy)
poly_ridge1_predy = ridge1.predict(poly_testx)
print("The r2 score after Polynomial ridge regression in svd is:",r2_score(testy,poly_ridge1_predy))

ridge2.fit(poly_trainx,trainy)
poly_ridge2_predy = ridge2.predict(poly_testx)
print("The r2 score after Polynomial ridge regression in cholesky is:",r2_score(testy,poly_ridge2_predy))

ridge3.fit(poly_trainx,trainy)
poly_ridge3_predy = ridge3.predict(poly_testx)
print("The r2 score after Bayesian Polynomial ridge regression is:",r2_score(testy,poly_ridge3_predy))

'''

'''

# Ridge Regression with Gradient Descent:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.linear_model import SGDRegressor, Ridge
from sklearn.datasets import load_diabetes

scaler = StandardScaler()
sgd = SGDRegressor(alpha=0.01,max_iter=10000,learning_rate="constant",penalty="l2",eta0=0.001,penalty='l2')
ridge1 = Ridge(alpha=0.01,solver='sparse_cg') # Not GD
ridge2 = Ridge(alpha=0.01,solver='sag')
ridge3 = Ridge(alpha=0.01,solver='saga')
ridge4 = Ridge(alpha=0.01,solver="lbfgs", positive=True)
ridge5 = Ridge(alpha=0.01, solver='lsqr') # Not GD
ridge6 = BayesianRidge(alpha_1=0.01, alpha_2=0.001, lambda_1=0.001, lambda_2=0.0001) # Uses a probabilistic approach

x,y = load_diabetes(return_X_y=True)
x = scaler.fit_transform(x)

trainx, testx, trainy, testy = train_test_split(x,y,test_size=0.3,random_state=33)

sgd.fit(trainx,trainy)
sgd_predy = sgd.predict(testx)
print("The R2 score after Stochastic Gradient Descent is:",r2_score(testy,sgd_predy))

ridge1.fit(trainx,trainy)
ridge1_predy = ridge1.predict(testx)
print("The R2 score after Ridge Regression with sparse_cg is:",r2_score(testy,ridge1_predy))

ridge2.fit(trainx,trainy)
ridge2_predy = ridge2.predict(testx)
print("The R2 score sfter Ridge regression with sag is:",r2_score(testy,ridge2_predy))

ridge3.fit(trainx,trainy)
ridge3_predy = ridge3.predict(testx)
print("The R2 score after Ridge Regression with saga is:",r2_score(testy,ridge3_predy))

ridge4.fit(trainx,trainy)
ridge4_predy = ridge4.predict(testx)
print("The R2 score after Ridge Regression with lbdgs is:",r2_score(testy,ridge4_predy))

ridge5.fit(trainx,trainy)
ridge5_predy = ridge5.predict(testx)
print("The R2 score after ridge regression with lsqr is:",r2_score(testy,ridge5_predy))

ridge6.fit(trainx,trainy)
ridge6_predy = ridge6.predict(testx)
print("The R2 score after using Bayesian ridge regression is:",r2_score(testy,ridge6_predy))

'''