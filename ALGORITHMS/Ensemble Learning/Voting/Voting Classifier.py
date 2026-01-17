import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier

tree = DecisionTreeClassifier()
svc = SVC(probability=True)
knn = KNeighborsClassifier()

x , y = load_iris(return_X_y=True, as_frame=True)

# estimators : list of ("str" , classifier) tuples

estimators = [("Decision tree",tree),("svc",svc),("knn",knn)]

'''

for estimator in estimators:
    z = cross_val_score(estimator=estimator[1],cv=10,X=x,y=y,scoring="accuracy")
    print(estimator[0],":",np.mean(z))

'''

vote1 = VotingClassifier(estimators=estimators,voting="hard") 
vote2 = VotingClassifier(estimators=estimators,voting="soft")

# It also has a weight parameter to apply different weights to more powerful classifiers

k = cross_val_score(vote1,x,y,cv=10,scoring="accuracy")
print("The cross val score after hard voting is:",np.mean(k))

l = cross_val_score(vote2,x,y,cv=10,scoring="accuracy")
print("The cross val score after soft voting is:",np.mean(l))

# We can also apply VotingClassifier on the same algorithms by tuning their parameters.