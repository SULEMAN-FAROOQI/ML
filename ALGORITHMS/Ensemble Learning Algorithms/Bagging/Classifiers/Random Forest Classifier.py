import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

x,y = load_iris(return_X_y=True, as_frame=True)

trainx , testx , trainy , testy = train_test_split(x,y,test_size=0.3,random_state=33)

forest1 = RandomForestClassifier(n_estimators=100, max_features="log2", max_samples=0.3, bootstrap=True)
forest2 = RandomForestClassifier(n_estimators=100, max_features="sqrt", max_samples=0.3, bootstrap=True)
forest3 = RandomForestClassifier(n_estimators=100, bootstrap=False)
forest4 = RandomForestClassifier(n_estimators=100, max_features="log2", max_samples=0.3, bootstrap=True, oob_score=True)
forest5 = RandomForestClassifier(n_estimators=100, max_features="sqrt", max_samples=0.3, bootstrap=True, oob_score=True)

forest1.fit(trainx,trainy)
forest1_predy = forest1.predict(testx)
print("The Accuracy Score after using Forest 1 is:",accuracy_score(testy,forest1_predy))

forest2.fit(trainx,trainy)
forest2_predy = forest2.predict(testx)
print("The Accuracy Score after using Forest 2 is:",accuracy_score(testy,forest2_predy))

forest3.fit(trainx,trainy)
forest3_predy = forest3.predict(testx)
print("The Accuracy Score after using Forest 3 is:",accuracy_score(testy,forest3_predy))

forest4.fit(trainx,trainy)
forest4_predy = forest4.predict(testx)
print("The Accuracy Score after using Forest 4 is:",accuracy_score(testy,forest4_predy))

forest5.fit(trainx,trainy)
forest5_predy = forest5.predict(testx)
print("The Accuracy Score after using Forest 5 is:",accuracy_score(testy,forest5_predy))

