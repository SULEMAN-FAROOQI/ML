import pandas as pd
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

x , y = load_diabetes(return_X_y=True, as_frame=True)

x = scaler.fit_transform(x)

trainx , testx, trainy , testy = train_test_split(x,y, test_size=0.3, random_state=33)

lr = LinearRegression()
boost1 = AdaBoostRegressor(estimator=lr, n_estimators=100, learning_rate=0.3)
boost2 = AdaBoostRegressor(estimator=lr, n_estimators=200, learning_rate=0.3)
boost3 = AdaBoostRegressor(estimator=None, n_estimators=100, learning_rate=0.3) 
boost4 = AdaBoostRegressor(estimator=None, n_estimators=200, learning_rate=0.3)

lr.fit(trainx, trainy)
lr_predy = lr.predict(testx)
print("The R2 Score after Linear Regression is:",r2_score(testy,lr_predy))

boost1.fit(trainx, trainy)
boost1_predy = boost1.predict(testx)
print("The R2 Score after Boost 1 is:",r2_score(testy,boost1_predy))

boost2.fit(trainx, trainy)
boost2_predy = boost2.predict(testx)
print("The R2 Score after Boost 2 is:",r2_score(testy,boost2_predy))

boost3.fit(trainx, trainy)
boost3_predy = boost3.predict(testx)
print("The R2 Score after Boost 3 is:",r2_score(testy,boost3_predy))

boost4.fit(trainx, trainy)
boost4_predy = boost4.predict(testx)
print("The R2 Score after Boost 4 is:",r2_score(testy,boost4_predy))
