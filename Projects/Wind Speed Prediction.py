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

m = XGBRegressor(n_estimators=248, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.75)

pipe = make_pipeline(z,m)

pipe.fit(trainx,trainy)
predy = pipe.predict(testx)
print("The R2 Score is:",r2_score(testy,predy))

