import numpy as np
import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder , FunctionTransformer, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

data = pd.read_csv("Datasets\\car_prediction_data.csv")
# print(data.sample(20))

x = data.drop("Present_Price", axis = 1)
y = data["Present_Price"]

trainx , testx , trainy , testy = train_test_split(x , y , test_size=0.3 , random_state=33)

def DataCleaner(x):
    x = x.copy()
    x["Car_Name"] = x["Car_Name"].str.lower().str.split(" ").str[0]
    x = x.drop("Selling_Price", axis=1)
    return x

f = FunctionTransformer(DataCleaner)

z = make_column_transformer(
        (OneHotEncoder(sparse_output=False, handle_unknown='ignore'), ["Car_Name", "Fuel_Type","Seller_Type", "Transmission"]),
        (StandardScaler(), ["Kms_Driven", "Year"]),
        remainder="passthrough"
)

m  = LinearRegression()

pipe = make_pipeline(f,z,m)

pipe.fit(trainx,trainy)
predy = pipe.predict(testx)
print("The R2 score after using Random Forest Regressor is:",r2_score(testy,predy))

data = {
    'Car_Name': ['Wagon R'],
    'Year': [2011],
    'Selling_Price': [2.85],
    'Kms_Driven': [5200],
    'Fuel_Type': ['Petrol'],
    'Seller_Type': ['Dealer'],
    'Transmission': ['Manual'],
    'Owner': [0]
}

new_data = pd.DataFrame(data)
prediction = pipe.predict(new_data)
print("Predicted Car value: " + str(prediction[0]), "Million $")
