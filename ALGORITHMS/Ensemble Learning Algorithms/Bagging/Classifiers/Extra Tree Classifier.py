import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

x,y = load_iris(return_X_y=True, as_frame=True) 

trainx , testx , trainy , testy = train_test_split(x,y,test_size=0.3,random_state=33)

etree1 = ExtraTreesClassifier(n_estimators=100, max_features="log2", max_samples=0.3, bootstrap=True)
etree2 = ExtraTreesClassifier(n_estimators=100, max_features="sqrt", max_samples=0.3, bootstrap=True)
etree3 = ExtraTreesClassifier(n_estimators=100, bootstrap=False)
etree4 = ExtraTreesClassifier(n_estimators=100, max_features="log2", max_samples=0.3, bootstrap=True, oob_score=True)
etree5 = ExtraTreesClassifier(n_estimators=100, max_features="sqrt", max_samples=0.3, bootstrap=True, oob_score=True)

etree1.fit(trainx,trainy)
etree1_predy = etree1.predict(testx)
print("The Accuracy Score after using Extra Trees 1 is:",accuracy_score(testy,etree1_predy))

etree2.fit(trainx,trainy)
etree2_predy = etree2.predict(testx)
print("The Accuracy Score after using Extra Trees 2 is:",accuracy_score(testy,etree2_predy))

etree3.fit(trainx,trainy)
etree3_predy = etree3.predict(testx)
print("The Accuracy Score after using Extra Trees 3 is:",accuracy_score(testy,etree3_predy))

etree4.fit(trainx,trainy)
etree4_predy = etree4.predict(testx)
print("The Accuracy Score after using Extra Trees 4 is:",accuracy_score(testy,etree4_predy))

etree5.fit(trainx,trainy)
etree5_predy = etree5.predict(testx)
print("The Accuracy Score after using Extra Trees 5 is:",accuracy_score(testy,etree5_predy))
