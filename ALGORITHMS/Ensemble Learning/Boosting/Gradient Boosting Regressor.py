import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

x,y = load_diabetes(return_X_y=True, as_frame=True)

gb1 = GradientBoostingRegressor(loss="absolute_error",n_estimators=150)
gb2 = GradientBoostingRegressor(loss="huber",n_estimators=150)
gb3 = GradientBoostingRegressor(loss="quantile",n_estimators=150)
gb4 = GradientBoostingRegressor(loss="squared_error",n_estimators=150)

trainx , testx, trainy , testy = train_test_split(x,y, test_size=0.3, random_state=33)

gb1.fit(trainx,trainy)
predy_gb1 = gb1.predict(testx)
print("The R2 Score after using absolute_error is:",r2_score(testy,predy_gb1))

gb2.fit(trainx,trainy)
predy_gb2 = gb2.predict(testx)
print("The R2 Score after using huber is:",r2_score(testy,predy_gb2))

gb3.fit(trainx,trainy)
predy_gb3 = gb3.predict(testx)
print("The R2 Score after using quantile is:",r2_score(testy,predy_gb3))

gb4.fit(trainx,trainy)
predy_gb4 = gb4.predict(testx)
print("The R2 Score after using squared_error is:",r2_score(testy,predy_gb4))
