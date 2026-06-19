import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import BaggingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

x , y = load_diabetes(return_X_y=True, as_frame=True)

trainx , testx , trainy, testy = train_test_split(x,y,test_size=0.3,random_state=33)

reg = LinearRegression() # We can use other classification Algorithms as well

# Row Sampling:

bag1 = BaggingRegressor(
    estimator=reg,
    n_estimators=500, # Num of Different variations in estimators
    max_samples=0.25, # Proportion of Sanple Data for Bootstraping
    bootstrap=True,
    random_state=33
)

bag1.fit(trainx,trainy)
bag1_predy = bag1.predict(testx)
print("The R2 score after Row sampling is:",r2_score(testy,bag1_predy))

# Pasting (Row Sampling without data repeation):

bag2 = BaggingRegressor(
    estimator=reg,
    n_estimators=500,
    max_samples=1.0,
    bootstrap=False,
    random_state=33
)

bag2.fit(trainx,trainy)
bag2_predy = bag2.predict(testx)
print("The R2 score after Pasting is:",r2_score(testy,bag2_predy))

# Column Sampling (Random Subspaces):

bag3 = BaggingRegressor(
    estimator=reg,
    n_estimators=500,
    max_samples=1.0,
    bootstrap=False,
    max_features=0.5, # Proportion of Feature Data for Bootstraping
    bootstrap_features=True,
    random_state=33
)

bag3.fit(trainx,trainy)
bag3_predy = bag3.predict(testx)
print("The R2 score after Column Sampling is:",r2_score(testy,bag3_predy))

# Row and Column sampling (Random Patches):

bag4 = BaggingRegressor(
    estimator=reg,
    n_estimators=500,
    max_samples=0.5,
    bootstrap=True,
    max_features=0.5,
    bootstrap_features=True,
    random_state=33
)

bag4.fit(trainx,trainy)
bag4_predy = bag4.predict(testx)
print("The R2 score after Both Row and Column sampling is:",r2_score(testy,bag4_predy))

# Out of Bag Data Evaluation:
# During Row Sampling statistically around 37 percent of sample data is not used, OOB parameter allows us to use these unused rows.

bag5 = BaggingRegressor(
    estimator=reg,
    n_estimators=500,
    max_samples=0.5,
    bootstrap=True,
    oob_score=True,
    random_state=33
)

bag5.fit(trainx,trainy)
bag5_predy = bag5.predict(testx)
print("The R2 score after using out of bag samples is:",r2_score(testy,bag5_predy))

# Bagging Without any Estimator:

bag6 = BaggingRegressor(estimator=None) # Estiamtor = Default Decision Trees

bag6.fit(trainx,trainy)
bag6_predy = bag6.predict(testx)
print("The R2 score without using as estimator is:",r2_score(testy,bag6_predy))

# Use Grid Search Cv for using Best Parameters.
