import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import VotingRegressor
from sklearn.linear_model import LinearRegression , Lasso , Ridge , ElasticNet , SGDRegressor

x,y = load_diabetes(return_X_y=True,as_frame=True)

lr = LinearRegression()
lasso  = Lasso(alpha=0.01,max_iter=100000000)
ridge = Ridge(alpha=0.01)
elastic = ElasticNet(alpha=0.01, l1_ratio=0.3, max_iter=100000000)
sgd = SGDRegressor(max_iter=10000,learning_rate="constant",random_state=None,eta0=0.01)

estimators = [("lr",lr),("sgd",sgd),("ridge",ridge),("lasso",lasso),("elasticnet",elastic)]

'''

for estimator in estimators:
    z = cross_val_score(estimator[1],x,y,cv=10,scoring="r2")
    print("The cross val score for",estimator[0],"is",np.mean(z))

'''
vote = VotingRegressor(estimators=estimators)
k = cross_val_score(vote,x,y,cv=10,scoring="r2")
print("The R2 score after cross validation with Voting Regressor is",np.mean(k))