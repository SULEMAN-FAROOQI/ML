import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from sklearn.impute import SimpleImputer

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
    data["GLD_ma5"]  = data["GLD"].shift(1).rolling(5).mean()
    data["GLD_ma20"] = data["GLD"].shift(1).rolling(20).mean()

    data["DATE"] = pd.to_datetime(data["Date"])
    data["day"]   = data["DATE"].dt.day
    data["month"] = data["DATE"].dt.month
    data["year"]  = data["DATE"].dt.year
    data = data.drop(columns=["Date", "DATE"])  
    return data

Data = ColumnTransformation(data)

x = Data.drop(columns = ["GLD"])
y = Data["GLD"]

trainx , testx , trainy , testy = train_test_split(x, y, test_size=0.3, shuffle=False) # For time series

z = make_column_transformer(
    (make_pipeline(SimpleImputer(strategy='median'), StandardScaler()), 
     ["GLD_lag1", "GLD_lag2", "GLD_lag3", "GLD_lag5",
      "SLV_lag1", "SLV_lag2", "GLD_ma5", "GLD_ma20"]),
    (StandardScaler(), ["SPX", "USO", "SLV", "EUR/USD", "day", "month", "year"]),
    remainder="drop"
)

m = Ridge(alpha=0.0144)

pipe = make_pipeline(z,m)

pipe.fit(trainx,trainy)
predy = pipe.predict(testx)
print("The R2 score after Ridge Regression is:",(r2_score(testy,predy)))

new_row = pd.DataFrame({
    "Date":    ["4/21/2015"],
    "SPX":     [2097.290039],
    "USO":     [19.450001],
    "SLV":     [15.320000],
    "EUR/USD": [1.074172],
    "GLD":     [np.nan]
})

combined = pd.concat([data, new_row], ignore_index=True)
combined = combined.sort_values('Date').reset_index(drop=True) 
combined = ColumnTransformation(combined)

new_input = combined.iloc[[-1]].drop("GLD", axis=1)
prediction = pipe.predict(new_input)
print("Predicted GLD price:", prediction[0],"$")