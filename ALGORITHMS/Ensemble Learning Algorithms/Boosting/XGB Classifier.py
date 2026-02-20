import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import xgboost as xgb

x,y = load_iris(return_X_y=True,as_frame=True)

trainx , testx , trainy, testy = train_test_split(x,y, test_size=0.3, random_state=33)

'''

XGBClassifier (Gradient Boosting): Builds trees sequentially to correct the errors of previous trees
by minimizing a classification loss function.

Common Hyperparameters for XGBClassifier:

1. n_estimators:
   The number of decision trees in the ensemble (boosting rounds).

2. learning_rate (eta):
   Controls how much each tree contributes to the model. Smaller values require more trees but
   improve generalization and reduce overfitting.

3. max_depth:
   The maximum depth of each decision tree. Higher values increase model complexity and the risk
   of overfitting.

4. subsample:
   The fraction of training samples used to train each tree. Values < 1.0 introduce randomness
   and help reduce overfitting.

5. colsample_bytree:
   The fraction of features (columns) used when building each tree. Reduces correlation between
   trees and improves generalization.

6. gamma:
   Minimum loss reduction required to make a split. Higher values make the model more conservative.

7. min_child_weight:
   Minimum sum of Hessians (second-order gradients) needed in a child node. Larger values prevent
   overfitting by avoiding splits on small or noisy data.

8. reg_lambda (L2 regularization):
   Controls L2 regularization on leaf weights. Increasing this makes the model more stable and
   smoother.

9. reg_alpha (L1 regularization):
   Controls L1 regularization on leaf weights. Encourages sparsity in leaf weights and can reduce
   the influence of less important features.

10. objective:
    Defines the classification loss function, for example:
    - 'binary:logistic'     → Binary classification (probability output)
    - 'multi:softprob'      → Multi-class classification (probability output)
    - 'multi:softmax'       → Multi-class classification (direct class labels)

11. scale_pos_weight:
    Balances positive and negative classes in imbalanced datasets (primarily for binary
    classification). Helps the model pay more attention to minority classes.

'''

xgbt = xgb.XGBClassifier(objective='multi:softprob', n_estimators=300, learning_rate=0.3, eval_metric='mlogloss')

xgbt.fit(trainx, trainy)
predy = xgbt.predict(testx)
print("The Accuracy score after using Extreme Gradient Boosting Classifier is:",accuracy_score(testy,predy))

'''

The most effective way to tune this model is the Trade-off Rule. If you decrease the learning_rate by 
half, you should roughly double the n_estimators. This allows the model to take smaller, 
more precise steps toward the minimum error.

'''