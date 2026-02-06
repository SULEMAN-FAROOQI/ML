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

Common Hyperparameters for XGBRFRegressor: 

1. n_estimators: The number of trees. Since they are parallel, more is generally better (until it plateaus), but 
it takes more memory.

2. max_depth: Controls how deep each tree can grow. Deeper trees capture more detail but are the #1 cause of overfitting.

3. learning_rate: In a true Random Forest, this is usually left at 1.0. Lowering it turns the model back into a 
"boosted" forest.

4. subsample: The fraction of rows used to train each tree. Lowering this adds "randomness" and prevents overfitting.

5. colsample_bynode: The fraction of features used for each split. This is vital for RF models to ensure trees aren't
all identical.

6. min_child_weight: Think of this as the minimum number of samples required to create a new leaf. 
Higher values make the model more conservative.

7. reg_alpha (L1) and reg_lambda (L2): These apply penalties to the weights of the leaves. 
If your features are noisy, increasing reg_lambda (default is 1) can help smooth out the predictions.

8. gamma: The minimum loss reduction required to make a further partition. It acts as a "pruning" mechanism.

'''

xgbrf = xgb.XGBRFRegressor(n_estimators=300, learning_rate = 1)

'''

Difference Between XGBRFRegressor and XGBRegressor:

1. XGBRegressor (Gradient Boosting): Builds trees sequentially to correct the errors (residuals) of 
previous trees (minimizes bias).

2. XGBRFRegressor (Random Forest): Builds trees in parallel (independently) and averages their predictions 
to reduce variance and prevent overfitting. 

'''

xgbrf.fit(trainx,trainy)
predy = xgbrf.predict(testx)
print("The R2 score after using Extreme Gradient Boosting Random Forest Regressor will be:",r2_score(testy,predy))
