import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import xgboost as xgb

housing = fetch_california_housing(as_frame=True)

x = housing.data
y = housing.target

scaler = StandardScaler()
x = scaler.fit_transform(x)

trainx , testx , trainy, testy = train_test_split(x,y, test_size=0.3, random_state=33)

'''

XGBRFRegressor builds multiple decision trees in parallel (bagging) and averages their predictions, 
similar to a Random Forest but using XGBoosts tree implementation.

Key Hyperparameters:

1. n_estimators:
   Number of trees built in parallel. More trees reduce variance but increase
   memory usage.

2. max_depth:
   Maximum depth of each tree. The primary driver of overfitting.

3. learning_rate:
   Should typically be left at 1.0. Unlike boosted XGBoost models, this does
   not control sequential learning.

4. subsample:
   Fraction of training rows used for each tree. Lower values increase
   randomness and reduce overfitting.

5. colsample_bynode:
   Fraction of features considered at each split. Essential for tree diversity.

6. min_child_weight:
   Minimum sum of Hessians required in a child node. Higher values make the model more stable.

7. reg_alpha (L1) and reg_lambda (L2):
   Regularization on leaf weights. Increasing reg_lambda helps smooth
   predictions when features are noisy.

8. gamma:
   Minimum loss reduction required to make a split, acting as a pruning mechanism.

'''

xgbrf = xgb.XGBRFRegressor(n_estimators=300, learning_rate = 1)

xgbrf.fit(trainx,trainy)
predy = xgbrf.predict(testx)
print("The R2 score after using Extreme Gradient Boosting Random Forest Regressor will be:",r2_score(testy,predy))
