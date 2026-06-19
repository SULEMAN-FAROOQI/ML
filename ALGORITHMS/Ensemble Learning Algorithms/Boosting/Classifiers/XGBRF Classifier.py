import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import xgboost as xgb

x,y = load_breast_cancer(return_X_y= True, as_frame=True)

trainx , testx , trainy, testy = train_test_split(x,y, test_size=0.3, random_state=33)

'''

XGBRFClassifier builds multiple decision trees in parallel (bagging) and averages their predictions, similar 
to a Random Forest but using XGBoosts tree implementation.

Key Hyperparameters:

1. n_estimators:
   Number of trees built in parallel. More trees reduce variance but
   increase memory usage.

2. max_depth:
   Maximum depth of each tree. Deeper trees increase risk of overfitting.

3. learning_rate:
   Typically left at 1.0. Does not affect boosting since trees are
   independent.

4. subsample:
   Fraction of training rows used for each tree. Lower values increase
   randomness and reduce overfitting.

5. colsample_bynode:
   Fraction of features considered at each split. Ensures tree diversity.

6. min_child_weight:
   Minimum sum of Hessians in a child node (depends on classification loss).
   Higher values make the model more conservative.

7. reg_alpha (L1) and reg_lambda (L2):
   Regularization on leaf weights. Helps smooth predictions with noisy features.

8. gamma:
   Minimum loss reduction required to make a split (pruning).

'''

xgbrf = xgb.XGBRFClassifier(n_estimators=300, learning_rate = 1)

xgbrf.fit(trainx,trainy)
predy = xgbrf.predict(testx)
print("The Accuracy score after using Extreme Gradient Boosting Random Forest Classifier will be:",accuracy_score(testy,predy))
