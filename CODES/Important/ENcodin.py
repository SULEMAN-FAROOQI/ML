# ORDINAL ENCODING:

'''

For encoding feature variable, we will use OrdianlEncoder and For encoding target variable, we will use LabelEncoder.

'''

# Example both Label Encoding and Ordinal Encoding:

'''

from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
import pandas as pd

dataframe = pd.read_csv("Datasets\\Encodin.csv")

oe = OrdinalEncoder(categories=[["HS", "UG", "PG", "MS"]])
dataframe["EDU"] = oe.fit_transform(dataframe[["Education"]])
dataframe.drop("Education", axis=1, inplace=True)

label = LabelEncoder()
dataframe["Place"] = label.fit_transform(dataframe["Placement"])
dataframe.drop("Placement", axis=1, inplace=True)

print(dataframe.head())

# Two square brackets are used because OrdinalEncoder takes data as a 2d input in the form of dataframe.

'''

# NOMINAL ENCODING:

'''

It is used to encode nominal feature variable.

'''

# Example One Hot Encoding:

'''

from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pandas as pd

dataframe = pd.read_csv("Datasets\\cars.csv")

hot = OneHotEncoder(drop="first",sparse_output=False, dtype = np.int32)
df = hot.fit_transform(dataframe[["fuel","owner"]])

counts = dataframe["brand"].value_counts()
repl = counts[counts <= 100].index
fd = pd.get_dummies(dataframe["brand"].replace(repl,"uncommon"))

d_f = np.hstack([fd.values,df,dataframe[["km_driven","selling_price"]].values])
print(d_f.shape)

'''

# Encoding Using Column Transformer:

'''

A ColumnTransformer is a single object that: 

1. Takes your feature matrix X (the input DataFrame),
2. Applies different transformations to different columns, and
3. Returns one combined NumPy array with all the transformed features side by side.

'''

'''

By using column transformer we can perform all types except Label encoding in one peice of code.

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder , OrdinalEncoder , LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

label = LabelEncoder()
df = pd.read_csv("Datasets\\covid_toy.csv")
# print(df.head())

Transformer = ColumnTransformer(transformers=[
    ("T1", SimpleImputer(), ["age"]),
    ("T2", OrdinalEncoder(categories=[["Mild","Strong"]],dtype=np.int32), ["cough"]),
    ("T3", OneHotEncoder(sparse_output=False, drop = "first", dtype= np.int32), ["gender","city"] )
], remainder="passthrough")

data = Transformer.fit_transform(df)
# print(data)

column = ["age", "gender", "cough-condition", "Delhi", "Mumbai", "Bangalore", "fever-temp", "has_covid"]
frame = pd.DataFrame(data=data, columns=column)
frame["covid-test"] = label.fit_transform(frame["has_covid"])
frame.drop("has_covid"  , axis=1 , inplace = True)

print(frame.head())

'''

# Algorithms where you SHOULD drop one column

'''

Linear Regression
Ridge Regression
Lasso Regression
ElasticNet
Logistic Regression
SGDRegressor / SGDClassifier
Generalized Linear Models (GLMs)

'''