import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

data = pd.read_csv("Datasets\\insurance.csv")
# print(data.describe())

x = data.drop("charges", axis = 1)
y = data["charges"]

trainx, testx, trainy, testy = train_test_split(x,y,test_size=0.2, random_state=42)

def ColumnTransformation(k):
    k = k.copy()
    k["BMI_Age"] = k["age"] * k["bmi"]
    k["Child_Age_ratio"] = k["age"] / (k["children"] + 1)
    return k

f = FunctionTransformer(ColumnTransformation)

z = make_column_transformer(
    (StandardScaler(), ["bmi"]),
    (OneHotEncoder(), ["sex", "smoker", "region"]),
    remainder="passthrough"
)

m = XGBRegressor(
    n_estimators=600,
    max_depth=3,
    learning_rate=0.0106,
    subsample=0.9175,
    colsample_bytree=0.9816,
    min_child_weight=5,
    reg_alpha=2.088,
    reg_lambda=9.998,
    random_state=42,
    verbosity=0
)

pipe = make_pipeline(f,z,m)

pipe.fit(trainx,trainy)
predy = pipe.predict(testx)
print("The R2 score is",r2_score(testy,predy))

print("------------------------------------------------------------------")

new_data = pd.DataFrame({
    'age':      [19],
    'sex':      ['female'],
    'bmi':      [27.9],
    'children': [0],
    'smoker':   ['yes'],
    'region':   ['southwest']
})

prediction = pipe.predict(new_data)
print("Predicted charge: $",prediction[0])