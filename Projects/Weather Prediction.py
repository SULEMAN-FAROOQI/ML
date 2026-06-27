import numpy as np
import pandas as pd
from sklearn.preprocessing import FunctionTransformer , StandardScaler , LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

label = LabelEncoder()

data = pd.read_csv("Datasets\\weather.csv")

# print(data.sample(20)) 
# print(data.describe())
# print(data.dtypes)

x = data.drop(columns = ["type"])
y = data["type"]

y = label.fit_transform(y)

trainx, testx, trainy, testy = train_test_split(x,y, test_size=0.3, random_state=33, stratify=y)

def ColumnTransformation(data):

    # Earth coordinates must be in radians for trigonometric functions
    latitude_rad = np.radians(data['latitude'])
    longitude_rad = np.radians(data['longitude'])

    # Project the flat coordinates onto a 3D sphere
    data['coord_x'] = np.cos(latitude_rad) * np.cos(longitude_rad)
    data['coord_y'] = np.cos(latitude_rad) * np.sin(longitude_rad)
    data['coord_z'] = np.sin(latitude_rad)

    data["DATE"] = pd.to_datetime(data["date_str"])
    data["day"]   = data["DATE"].dt.day
    data["month"] = data["DATE"].dt.month
    data["year"]  = data["DATE"].dt.year
    data = data.drop(columns=["date_str", "DATE", "longitude", "latitude", "serialid", "id", "station_name"]) 
    data = data.dropna() 
    return data

f = FunctionTransformer(ColumnTransformation)

z = make_column_transformer(
    (StandardScaler(), ["degrees_from_mean" , "max_temp", "min_temp", "coord_x", "coord_y", "coord_z", "day", "month", "year"]),
    remainder = "drop"
)

m = LGBMClassifier(
    n_estimators      = 155,
    max_depth         = 9,
    learning_rate     = 0.064,
    num_leaves        = 117,
    min_child_samples = 50,
    subsample         = 0.573,
    colsample_bytree  = 0.862,
    reg_alpha         = 4.88e-7,
    reg_lambda        = 6.18e-7,
    random_state      = 42,
    n_jobs            = -1,
    class_weight="balanced",
    verbose = -1
)

pipe  = make_pipeline(f,z,m)

pipe.fit(trainx,trainy)
predy = pipe.predict(testx)

print("The Accuracy is:",accuracy_score(testy,predy))

print("--------------------------------------------------------")

new_data = pd.DataFrame({
    "date_str":          ["2002-01-26"],
    "degrees_from_mean": [14.69],
    "id":                ["USW00024029"],
    "longitude":         [-106.9689],
    "latitude":          [44.7694],
    "max_temp":          [16.1],
    "min_temp":          [-7.2],
    "station_name":      ["SHERIDAN CO AP"],
    "serialid":          [354415]
})

prediction = pipe.predict(new_data)
print("The Residents of",new_data["station_name"].values[0],"experienced",label.inverse_transform(prediction)[0],"weather on",new_data["date_str"].values[0])
