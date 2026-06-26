import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_colwidth', 100)

data = pd.read_csv("Datasets\\gld_price_data.csv")
# print(data.head())
# print(data.describe())

def ColumnTransformation(data):
    data["GLD_lag1"]  = data["GLD"].shift(1)   # yesterday's GLD price
    data["GLD_lag2"]  = data["GLD"].shift(2)   # 2 days ago GLD price
    data["GLD_lag3"]  = data["GLD"].shift(3)   # 3 days ago GLD price
    data["GLD_lag5"]  = data["GLD"].shift(5)   # 5 days ago GLD price
    data["SLV_lag1"]  = data["SLV"].shift(1)   # yesterday's SLV price
    data["SLV_lag2"]  = data["SLV"].shift(2)   # 2 days ago SLV price
    data["GLD_ma5"]   = data["GLD"].rolling(5).mean()    # average of last 5 days GLD
    data["GLD_ma20"]  = data["GLD"].rolling(20).mean()   # average of last 20 days GLD

    data["DATE"] = pd.to_datetime(data["Date"])
    data["day"]   = data["DATE"].dt.day
    data["month"] = data["DATE"].dt.month
    data["year"]  = data["DATE"].dt.year
    data = data.drop(columns=["Date", "DATE"])  
    data = data.dropna()
    return data

Data = ColumnTransformation(data)

x = Data.drop(columns = ["GLD"])
y = Data["GLD"]

trainx , testx , trainy , testy = train_test_split(x, y, test_size=0.3, random_state=33, shuffle=False) # For time series

z = make_column_transformer(
    (StandardScaler(), ["SPX", "USO", "SLV", "EUR/USD",
                        "GLD_lag1", "GLD_lag2", "GLD_lag3", "GLD_lag5",
                        "SLV_lag1", "SLV_lag2",
                        "GLD_ma5", "GLD_ma20",
                        "day", "month", "year"]),  
    remainder="drop"
)

m = Ridge(alpha=0.0144)

pipe = make_pipeline(z,m)

pipe.fit(trainx,trainy)
predy = pipe.predict(testx)
print("The R2 score after Ridge Regression is:",(r2_score(testy,predy)))

new_data = pd.DataFrame({
    "Date": ["1/2/2008"],
    "SPX":[1447.160034],
    "USO":[78.470001],
    "SLV":[15.18],
    "EUR/USD":[1.471692],
    "GLD_lag1":[84.86],
    "GLD_lag2":[84.70],
    "GLD_lag3":[84.55],
    "GLD_lag5":[84.20],
    "SLV_lag1":[15.10],
    "SLV_lag2":[15.05],
    "GLD_ma5":[84.60],
    "GLD_ma20":[83.90],
    "day":[2],
    "month":[1],
    "year":[2008]
})

prediction = pipe.predict(new_data) 
print("Predicted GLD price:", prediction[0],"$")
