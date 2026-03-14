import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder , StandardScaler

data = pd.read_csv("C:\\Users\\Suleman\\Desktop\\Workspace\\import\\DOCUMENTS\\Codes\\ML\\Datasets\\example.csv")
dataset = pd.DataFrame(data)

label = LabelEncoder()
scaler = StandardScaler()

dataset["Status"] = label.fit_transform(dataset["Placement_Status"])
dataset.drop("Placement_Status", axis=1 , inplace=True)
dataset.drop("Secondary_Percentage", axis=1 , inplace=True)
dataset.drop_duplicates()

x = dataset.iloc[: , : -1]
y = dataset.iloc[: , -1]

trainx , testx, trainy , testy = train_test_split(x,y, test_size=0.3, random_state=3)

trainx = scaler.fit_transform(trainx)
testx = scaler.transform(testx)

param_grid = {
    "C": [0.01, 0.1, 1, 10, 100],
    "penalty": ["l2"],
    "solver": ["lbfgs", "liblinear"],
    "max_iter": [1000, 2000]
}

grid = GridSearchCV(
    LogisticRegression(),
    param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

grid.fit(trainx, trainy)

best_model = grid.best_estimator_

print("Best Params:", grid.best_params_)
print("CV Accuracy:", grid.best_score_)
print("Test Accuracy:", accuracy_score(testy, best_model.predict(testx)))

# For Regression with Pipeline:

'''

import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_diabetes

x , y = load_diabetes(return_X_y=True, as_frame=True)

pipeline = Pipeline([
    ("scaler", StandardScaler()), 
    ("xgb", xgb.XGBRegressor())
])

# Define the hyperparameter grid for GridSearchCV Search
# Double underscores (__) are used to target the parameters of 'xgb' in the pipeline
param = {
    'xgb__subsample': [0.05,0.1,0.5,1],
    'xgb__max_depth': [2,3,5,8],
    'xgb__colsample_bytree': [0.05,0.1,0.5,1]
}

grid = GridSearchCV(
    estimator=pipeline,
    param_grid=param, 
    scoring='r2', 
    cv=4,
    n_jobs=-1
)

# Fit the model to the data
grid.fit(x, y)

# Optional: Print the best score and parameters found
print(f"Best Score: {grid.best_score_}")
print(f"Best Parameters: {grid.best_params_}")

'''