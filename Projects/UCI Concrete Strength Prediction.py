import numpy as np 
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from catboost import CatBoostRegressor

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_colwidth', 100)

data = pd.read_csv('Datasets\\concrete_data.csv')
# print(data.sample(10))
# print(data.corr()["Strength"])

x = data.drop("Strength", axis = 1)
y = data["Strength"]



trainx, testx, trainy, testy = train_test_split(x, y, test_size=0.3, random_state=33)

def FeatureTransformation(df):

    df = df.copy()
    df["Total Binder"] = df["Cement"] + df["Blast Furnace Slag"] + df["Fly Ash"]
    df["Water-Binder ratio"] = df["Water"] / df["Total Binder"]
    df["Superplasticizer-binder ratio"] = df["Superplasticizer"] / df["Total Binder"]
    df["aggregate ratio"] = df["Coarse Aggregate"] / df["Fine Aggregate"]
    df["Interaction term"] = df["Cement"] / df["Water-Binder ratio"] # approximates the classic Abrams Law relationship (strength ∝ cement/water)
    df["Total Aggregate"] = df["Coarse Aggregate"] + df["Fine Aggregate"]
    df["Age (log)"] = np.log1p(df["Age"])  # linearizes strength-vs-age curing curve
    df["Slag-Binder ratio"] = df["Blast Furnace Slag"] / df["Total Binder"]
    df["FlyAsh-Binder ratio"] = df["Fly Ash"] / df["Total Binder"]

    return df

f = FunctionTransformer(FeatureTransformation)
f.set_output(transform="pandas")

z = make_column_transformer(
    (StandardScaler(), ["Cement", "Blast Furnace Slag", "Fly Ash", "Water", "Superplasticizer",
              "Coarse Aggregate", "Fine Aggregate", "Age", "Total Binder",
              "Water-Binder ratio", "Superplasticizer-binder ratio",
              "aggregate ratio", "Interaction term",
              "Total Aggregate", "Age (log)", "Slag-Binder ratio", "FlyAsh-Binder ratio"]),
    remainder="passthrough"
)

m = CatBoostRegressor(iterations=245, depth=8, learning_rate=0.0988, l2_leaf_reg=0.651, allow_writing_files=False, silent=True)

pipe = make_pipeline(f,z,m)

pipe.fit(trainx, trainy)
predy = pipe.predict(testx)

print("The R2 score is:",r2_score(testy,predy))

new_data = pd.DataFrame([{
    "Cement": 310.0,
    "Blast Furnace Slag": 100.0,
    "Fly Ash": 0.0,
    "Water": 180.0,
    "Superplasticizer": 6.0,
    "Coarse Aggregate": 980.0,
    "Fine Aggregate": 750.0,
    "Age": 28
}])

pred = pipe.predict(new_data)
print("predicted strength:", round(pred[0], 3),"MPa")