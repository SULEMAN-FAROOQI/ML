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

Common Hyperparameters for XGBRegressor:

1. n_estimators: The number of decision trees in the ensemble (boosting rounds).

2. learning_rate (eta): Controls how much the model learns in each step (shrinkage), smaller values 
prevent overfitting.

3. max_depth: The maximum depth of each decision tree; a higher value increases model complexity and 
overfitting.

4. subsample: The fraction of samples used to train each tree.

5. colsample_bytree: The fraction of features (columns) used when building each tree. 

6. gamma: The "complexity control." A tree split only happens if the resulting loss reduction exceeds 
this value. High gamma = conservative model.

7. min_child_weight: Defines the minimum "strength" or number of samples required in a leaf. 
In regression, it relates to the sum of instances.

8. reg_lambda (L2): The default is 1. Increasing this makes leaf weights smaller and smoother.

9. reg_alpha (L1): Useful if you have a massive amount of features and want the model to ignore 
the useless ones (it can push weights to zero).

'''

xgbt = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=300, learning_rate=0.3)

'''

Difference Between XGBRFRegressor and XGBRegressor:

1. XGBRegressor (Gradient Boosting): Builds trees sequentially to correct the errors (residuals) of 
previous trees (minimizes bias).

2. XGBRFRegressor (Random Forest): Builds trees in parallel (independently) and averages their predictions 
to reduce variance and prevent overfitting. 

'''

xgbt.fit(trainx, trainy)
predy = xgbt.predict(testx)
print("The R2 score after using Extreme Gradient Boosting Regressor is:",r2_score(testy,predy))

'''

The most effective way to tune this model is the Trade-off Rule. If you decrease the learning_rate by 
half, you should roughly double the n_estimators. This allows the model to take smaller, 
more precise steps toward the minimum error.

'''