import numpy as np
import pandas as pd
from sklearn.preprocessing import FunctionTransformer , StandardScaler , OrdinalEncoder , LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.metrics import accuracy_score
from sklearn.ensemble import ExtraTreesRegressor , ExtraTreesClassifier
from sklearn.model_selection import train_test_split , cross_val_score
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv("Datasets\\penguins.csv")

# print(df.sample(20))
# print(df.describe())
# print(df.isnull().sum())

label = LabelEncoder()

x = df.drop("species" , axis = 1)
y = df["species"]
y = label.fit_transform(y)

trainx, testx, trainy, testy = train_test_split(x,y, test_size=0.3, random_state=33, stratify=y)

def DataTransformer(data):
    data['bills (Len/Depth)'] = df['bill_length_mm'] / df['bill_depth_mm']
    data = data.drop(columns = ["bill_length_mm", "bill_depth_mm", "Unnamed: 0"])
    return data

f = FunctionTransformer(DataTransformer)

z = make_column_transformer(
    (OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=np.nan, encoded_missing_value=np.nan) , ["island", "sex"]),
    remainder = "passthrough"
)
z.set_output(transform="pandas")

i = IterativeImputer(max_iter=10, imputation_order="ascending", estimator=ExtraTreesRegressor(n_estimators=200, random_state=42, n_jobs=-1))
i.set_output(transform="pandas")

l = make_column_transformer(
    (StandardScaler() , [3,4,5]),
    remainder="passthrough"
)

m = ExtraTreesClassifier(n_estimators=500, random_state=33)

pipe = make_pipeline(f,z,i,l,m)

pipe.fit(trainx,trainy)
predy = pipe.predict(testx)
print("The Accuracy score is:",accuracy_score(testy,predy))

test_penguin = pd.DataFrame({
    "Unnamed: 0": [345],
    "island": ["Biscoe"],
    "bill_length_mm": [49.5],
    "bill_depth_mm": [15.0],
    "flipper_length_mm": [221],
    "body_mass_g": [5650],
    "sex": ["Male"],
    "year": [2008]
})

prediction = pipe.predict(test_penguin)
result = (
    "Adelie" if prediction[0] == 0
    else "Chinstrap" if prediction[0] == 1
    else "Gentoo"
)

print("Specimen",test_penguin["Unnamed: 0"].values[0],"is classified as a",result)