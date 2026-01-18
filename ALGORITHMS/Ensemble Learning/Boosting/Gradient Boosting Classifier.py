import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

x,y = load_iris(return_X_y=True, as_frame=True)

gb1 = GradientBoostingClassifier(loss="log_loss",n_estimators=100, criterion="friedman_mse")
gb2 = GradientBoostingClassifier(loss="log_loss",n_estimators=100, criterion="squared_error")

# log loss = exponential is only applicable for binary.
# log loss = deviance has been dated into log loss

trainx , testx, trainy , testy = train_test_split(x,y, test_size=0.3, random_state=33)

gb1.fit(trainx,trainy)
predy_gb1 = gb1.predict(testx)
print("The Accuracy Score after using friedman mse is:",accuracy_score(testy,predy_gb1))

gb2.fit(trainx,trainy)
predy_gb2 = gb2.predict(testx)
print("The Accuracy Score after using squared_error is:",accuracy_score(testy,predy_gb2))
