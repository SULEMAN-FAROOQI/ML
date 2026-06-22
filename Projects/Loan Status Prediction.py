import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder , StandardScaler , LabelEncoder , FunctionTransformer , MinMaxScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from lightgbm import LGBMClassifier
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

hot = LabelEncoder()

df = pd.read_csv("Datasets\\loan.csv")
# print(df.head())
# print(df.isnull().sum())
# print(df.describe())

x = df.drop("Loan_Status" , axis = 1)
y = df["Loan_Status"]

y = hot.fit_transform(y)

trainx , testx , trainy , testy = train_test_split(x, y, test_size=0.3, random_state=33)

def FeatureTransformation(data):
    data = data.drop("Loan_ID", axis = 1)
    return data

r = FunctionTransformer(FeatureTransformation)

z = make_column_transformer(
    (OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=np.nan, encoded_missing_value=np.nan) , ["Gender", "Married", "Education", "Self_Employed", "Property_Area", "Dependents"]),
    remainder = "passthrough"
)
z.set_output(transform="pandas")

i = IterativeImputer(max_iter=10, imputation_order="ascending", estimator=ExtraTreesRegressor())
i.set_output(transform="pandas")

k = make_column_transformer(
    (StandardScaler() , [7,8,9]),
    (MinMaxScaler() , [10]),
    remainder="passthrough"
)

m = LGBMClassifier(
    n_estimators=380,
    max_depth=7,
    learning_rate=0.258,
    num_leaves=34,
    min_child_samples=27,
    subsample=0.919,
    colsample_bytree=0.514,
    reg_alpha=9.19,
    reg_lambda=1.71,
    random_state=42
)

pipe = make_pipeline(r,z,i,k,m)

import os, contextlib
with open(os.devnull, 'w') as devnull:
    with contextlib.redirect_stdout(devnull):
        pipe.fit(trainx, trainy)

predy = pipe.predict(testx)
print("The Accuracy Score is:",accuracy_score(testy,predy))

test_data = pd.DataFrame({
    'Loan_ID':            ['LP999002'],
    'Gender':             ['Male'],
    'Married':            ['No'],
    'Dependents':         ['3+'],
    'Education':          ['Not Graduate'],
    'Self_Employed':      ['Yes'],
    'ApplicantIncome':    [2200],
    'CoapplicantIncome':  [0.0],
    'LoanAmount':         [None], 
    'Loan_Amount_Term':   [360.0],
    'Credit_History':     [0.0],
    'Property_Area':      ['Rural'],
})

predictions = pipe.predict(test_data)
result = "Approved" if predictions[0] == 1 else "Rejected"
print("Loan ID:",test_data["Loan_ID"].values[0],"request is",result)