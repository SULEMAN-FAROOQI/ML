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

XGBRegressor (Gradient Boosting): Builds trees sequentially to correct the errors (residuals) of
previous trees by minimizing a regression loss function.

Common Hyperparameters for XGBRegressor:

1. n_estimators:
   The number of decision trees in the ensemble (boosting rounds).

2. learning_rate (eta):
   Controls how much the model learns in each step (shrinkage). Smaller values require more trees
   but improve generalization and reduce overfitting.

3. max_depth:
   The maximum depth of each decision tree. Higher values increase model complexity and risk of
   overfitting.

4. subsample:
   The fraction of training samples used to train each tree. Values < 1.0 add randomness and help
   reduce overfitting.

5. colsample_bytree:
   The fraction of features (columns) used when building each tree.

6. gamma:
   Minimum loss reduction required to make a split. Higher values make the model more conservative.

7. min_child_weight:
   Minimum sum of Hessians (second-order gradients) needed in a child node. Larger values prevent
   overfitting by avoiding splits on small or noisy data.

8. reg_lambda (L2 regularization):
   Controls L2 regularization on leaf weights. Increasing this makes the model more robust and
   smoother.

9. reg_alpha (L1 regularization):
   Controls L1 regularization on leaf weights. Encourages sparsity in leaf weights and can reduce
   the influence of less important features.

10. objective:
    Specifies the regression loss function:
    - 'reg:squarederror'      → Mean Squared Error (default)
    - 'reg:absoluteerror'    → Mean Absolute Error
    - 'reg:pseudohubererror' → Robust loss (less sensitive to outliers)

'''

xgbt = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=300, learning_rate=0.3)

xgbt.fit(trainx, trainy)
predy = xgbt.predict(testx)
print("The R2 score after using Extreme Gradient Boosting Regressor is:",r2_score(testy,predy))

'''

The most effective way to tune this model is the Trade-off Rule. If you decrease the learning_rate by 
half, you should roughly double the n_estimators. This allows the model to take smaller, 
more precise steps toward the minimum error.

'''