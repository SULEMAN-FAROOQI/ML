import numpy as np
import pandas as pd
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from lightgbm import LGBMRegressor
import warnings
warnings.filterwarnings('ignore')

data = pd.read_csv("Datasets\\exercise.csv")
# print(data.sample(10))
# print(data.isnull().sum())
# print(data.describe())

x = data.drop("Calories", axis = 1)
y = data["Calories"]

trainx, testx, trainy, testy = train_test_split(x, y, test_size=0.3, random_state=33)

def ColumnTransformation(df):

    df["BMI"] = df["Weight"] / ((df["Height"] / 100) ** 2)
    df["Workout_Intensity"] = df["Heart_Rate"] * df["Body_Temp"]
    df["Heart_Rate_Intensity"] = df["Heart_Rate"] / (220 - df["Age"])

    df = df.drop(columns = ["User_ID","Calories_x","Calories_y"])
    return df

f = FunctionTransformer(ColumnTransformation)
f.set_output(transform="pandas")

z = make_column_transformer(
    (OneHotEncoder() , ["Gender"]), 
    remainder="passthrough"
)

m = LGBMRegressor(
    n_estimators      = 987,
    max_depth         = 4,
    learning_rate     = 0.1492,
    num_leaves        = 95,
    min_child_samples = 11,
    subsample         = 0.8475,
    colsample_bytree  = 0.7118,
    reg_alpha         = 0.5842,
    reg_lambda        = 0.0019,
    random_state      = 42,
    verbose           = -1
)

pipe = make_pipeline(f,z,m)

pipe.fit(trainx,trainy)
predy = pipe.predict(testx)
print("The R2 score is:",r2_score(testy,predy))

print("---------------------------------------------------------")

sample = pd.DataFrame([{
    "User_ID"   : 14733363,
    "Gender"    : "male",
    "Age"       : 68,
    "Height"    : 190.0,
    "Weight"    : 94.0,
    "Duration"  : 29.0,
    "Heart_Rate": 105.0,
    "Body_Temp" : 40.8,
    "Calories_x": 231.0,
    "Calories_y": 231.0
}])

prediction = pipe.predict(sample)
print("User:",sample["User_ID"].values[0],"burned",prediction[0],"calories in the duration of",sample["Duration"].values[0],"minutes.")