import pandas as pd
import numpy as np
from sklearn.datasets import load_wine
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

x , y = load_wine(return_X_y=True, as_frame=True)

trainx , testx, trainy , testy = train_test_split(x,y, test_size=0.3, random_state=33)

boost1 = AdaBoostClassifier(estimator=LogisticRegression(solver="saga", max_iter=1000, C=1.0), n_estimators=100, learning_rate=0.3)
boost2 = AdaBoostClassifier(estimator=LogisticRegression(solver="saga", max_iter=1000, C=1.0), n_estimators=200, learning_rate=0.3)
boost3 = AdaBoostClassifier(estimator=None, n_estimators=100, learning_rate=0.3)
boost4 = AdaBoostClassifier(estimator=None, n_estimators=200, learning_rate=0.3)
logrg = LogisticRegression(solver="saga")

logrg.fit(trainx, trainy)
logrg_predy = logrg.predict(testx)
print("The Accuracy Score after Logistic Regression is:",accuracy_score(testy,logrg_predy))

boost1.fit(trainx, trainy)
boost1_predy = boost1.predict(testx)
print("The Accuracy Score after Boost 1 is:",accuracy_score(testy,boost1_predy))

boost2.fit(trainx, trainy)
boost2_predy = boost2.predict(testx)
print("The Accuracy Score after Boost 2 is:",accuracy_score(testy,boost2_predy))

boost3.fit(trainx, trainy)
boost3_predy = boost3.predict(testx)
print("The Accuracy Score after Boost 3 is:",accuracy_score(testy,boost3_predy))

boost4.fit(trainx, trainy)
boost4_predy = boost4.predict(testx)
print("The Accuracy Score after Boost 4 is:",accuracy_score(testy,boost4_predy))

# ADA Boost works good on weak algorithms where logistic regression is a strong algorithm so its accuracy is 
# slightly decreased where C is convergence.