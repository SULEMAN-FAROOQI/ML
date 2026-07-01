import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split 
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.metrics import r2_score
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')

data = pd.read_csv("Datasets\\wind_dataset.csv")

def ColumnTransformer(data):
    data["WIND_lag1"] = data["WIND"].shift(1)
    data["WIND_lag3"] = data["WIND"].shift(3)

    data["WIND_ma5"]  = data["WIND"].shift(1).rolling(5).mean()
    data["WIND_ma10"] = data["WIND"].shift(1).rolling(10).mean()

    data["year"] = pd.to_datetime(data["DATE"]).dt.year
    data["dayofyear"] = pd.to_datetime(data["DATE"]).dt.dayofyear
    data = data.drop(columns = ["DATE"])
    return data

Data = ColumnTransformer(data)

x = Data.drop(columns = ["WIND"])
y = Data["WIND"]

trainx ,testx, trainy , testy = train_test_split(x, y, test_size=0.3, random_state=33)

z = make_column_transformer(
    (SimpleImputer(strategy="median") , ["IND.1","T.MAX", "IND.2", "T.MIN", "T.MIN.G"]),
    remainder="passthrough"
)

m = XGBRegressor(
    n_estimators=458,
    max_depth=8,
    learning_rate=0.0074,
    subsample=0.586,
    colsample_bytree=0.957,
    min_child_weight=7,
    gamma=0.215,
    reg_alpha=0.00046,
    reg_lambda=0.418,
    random_state=42,
    verbosity=0
)

pipe = make_pipeline(z,m)

pipe.fit(trainx,trainy)
predy = pipe.predict(testx)
print("The R2 Score is:",r2_score(testy,predy))

new_data = pd.DataFrame({
    'IND'      : [0],
    'RAIN'     : [5.8],
    'IND.1'    : [0.0],
    'T.MAX'    : [16.1],
    'IND.2'    : [0.0],
    'T.MIN'    : [3.7],
    'T.MIN.G'  : [-0.2],
    'WIND_lag1': [5.21],
    'WIND_lag3': [8.71],
    'WIND_ma5' : [6.842],
    'WIND_ma10': [7.188],
    'year'     : [1961],
    'dayofyear': [101]
})

prediction = pipe.predict(new_data)
print("The Speed of wind is:",prediction[0])