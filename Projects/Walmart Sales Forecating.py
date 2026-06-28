import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor
from sklearn.impute import SimpleImputer
import warnings
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_colwidth', 100)

data = pd.read_csv("Datasets\\Walmart_Sales.csv")

# print(data.sample(10))
# print("-----------------------------------------------------------------------------------------------------")
# print(data.describe())

def DataTransformation(df):

    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    df['Week'] = df['Date'].dt.isocalendar().week.astype(int)
    df['Month'] = df['Date'].dt.month
    df['Year'] = df['Date'].dt.year

    df = df.sort_values(['Store', 'Date']).reset_index(drop=True)
    df['Purchasing_Power_Index'] = 1 / (df['CPI'] * df['Unemployment'])
    df = df.drop("Date", axis = 1)

    df['Sales_lag_1'] = df.groupby('Store')['Weekly_Sales'].shift(1)   # last week's sales for this store
    df['Sales_lag_2'] = df.groupby('Store')['Weekly_Sales'].shift(2)   # 2 weeks ago sales for this store
    df['Sales_lag_4'] = df.groupby('Store')['Weekly_Sales'].shift(4)   # 1 month ago sales for this store
    df['Sales_lag_8'] = df.groupby('Store')['Weekly_Sales'].shift(8)   # 2 months ago sales for this store
    df['Sales_lag_52'] = df.groupby('Store')['Weekly_Sales'].shift(52) # same week last year for this store

    df['Sales_rolling_4'] = df.groupby('Store')['Weekly_Sales'].transform(lambda x: x.shift(1).rolling(4).mean())  # avg sales over last 4 weeks for this store
    df['Sales_rolling_8'] = df.groupby('Store')['Weekly_Sales'].transform(lambda x: x.shift(1).rolling(8).mean())  # avg sales over last 8 weeks for this store
        # transform keeps rolling inside each store

    return df

Data = DataTransformation(data)
# print(Data.sample(10))

x = Data.drop("Weekly_Sales", axis = 1)
y = Data["Weekly_Sales"]

trainx, testx, trainy, testy = train_test_split(x,y, test_size=0.2, shuffle=False)

z = make_column_transformer(
    (make_pipeline(SimpleImputer(strategy='median'), StandardScaler()),['Sales_lag_1', 'Sales_lag_2', 'Sales_lag_4', 'Sales_lag_8', 'Sales_lag_52',
      'Sales_rolling_4', 'Sales_rolling_8']),
    (StandardScaler(), ['Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'Purchasing_Power_Index']),
    remainder='passthrough'
)

m = LGBMRegressor(
    n_estimators=702,
    learning_rate=0.0258,
    num_leaves=115,
    max_depth=4,
    min_child_samples=37,
    subsample=0.7857,
    colsample_bytree=0.5008,
    reg_alpha=0.0722,
    reg_lambda=0.0277,
    verbose=-1
)

pipe = make_pipeline(z,m)

pipe.fit(trainx,trainy)
predy = pipe.predict(testx)
print("The R2 Score is:",r2_score(testy,predy))

print("-----------------------------------------------------------------------------------------------------")

new_row = pd.DataFrame({
    "Store": [1],
    "Date": ["12-02-2011"], 
    "Weekly_Sales": [None],   
    "Holiday_Flag": [0],
    "Temperature": [42.27],
    "Fuel_Price": [2.989],
    "CPI": [212.566881],
    "Unemployment": [7.742]
})

combined = pd.concat([data, new_row], ignore_index=True)
combined = combined.sort_values(['Store', 'Date']).reset_index(drop=True) 
combined = DataTransformation(combined)

new_input = combined.iloc[[-1]].drop("Weekly_Sales", axis=1)
prediction = pipe.predict(new_input)
print("The Week Sales of Walmart store",new_input["Store"].values[0],"is",prediction[0],"$")
