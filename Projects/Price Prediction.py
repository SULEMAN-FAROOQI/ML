# This code represnets an example of a ML workflow. It predicts the median house value based on various features of the housing data. 
# The workflow includes data preprocessing, feature engineering, model training, and evaluation using R2 score and cross-validation. 
# Finally, it demonstrates how to make a prediction for a new input data point.

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from xgboost import XGBRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score, train_test_split

data = pd.read_csv('Datasets/housing.csv')

x = data.drop("median_house_value", axis=1)
y = data["median_house_value"]

trainx, testx, trainy, testy = train_test_split(x, y, test_size=0.2, random_state=42)

i = make_column_transformer(
    (SimpleImputer(strategy='median'), ['longitude', 'latitude', 'housing_median_age',
                                  'total_rooms', 'total_bedrooms', 'population',
                                  'households', 'median_income']),
     (SimpleImputer(strategy='most_frequent'), ['ocean_proximity']),
    remainder='passthrough'  
)
i.set_output(transform="pandas")

def feature_transformation(data):
    data = data.copy()
    data.columns = [col.split('__')[-1] for col in data.columns]
    data['bedrooms_per_room']        = data['total_bedrooms'] / data['total_rooms']
    data['population_per_household'] = data['population'] / data['households']
    data['income_per_household']     = data['median_income'] / data['households']
    data.drop(columns=['total_bedrooms', 'total_rooms',
                       'population', 'households', 'median_income'], inplace=True)
    return data

f = FunctionTransformer(feature_transformation)

z = make_column_transformer(
    (StandardScaler(), ['longitude', 'latitude', 'housing_median_age',
                        'bedrooms_per_room', 'population_per_household',
                        'income_per_household']),
    (OneHotEncoder(), ['ocean_proximity']),
    remainder='passthrough'
)

m = XGBRegressor(
    objective        = 'reg:squarederror',
    n_estimators     = 700,
    max_depth        = 6,
    learning_rate    = 0.05,
    subsample        = 0.7,
    colsample_bytree = 0.8,
    min_child_weight = 5,
    reg_alpha        = 0.1,   
    reg_lambda       = 1.5,   
    gamma            = 0.1,
    tree_method      = 'hist',
    random_state     = 42
)

pipeline = make_pipeline(i, f, z, m)

pipeline.fit(trainx,trainy)
predy = pipeline.predict(testx)
print("The accuracy score is: " +str(r2_score(testy,predy)))
print("After cross validation our score is: " +str(np.mean(cross_val_score(pipeline, trainx ,trainy , cv=10, scoring='r2'))))

test_input = pd.DataFrame({
    'longitude': [-122.23],
    'latitude': [37.88],
    'housing_median_age': [41.0],
    'total_rooms': [880.0],
    'total_bedrooms': [129.0],
    'population': [322.0],
    'households': [126.0],
    'median_income': [8.3252],
    'ocean_proximity': ['NEAR BAY']
})

prediction = pipeline.predict(test_input)
print("Predicted house value: $" + str(round(prediction[0])))