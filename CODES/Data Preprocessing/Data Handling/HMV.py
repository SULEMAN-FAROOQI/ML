'''

This code offers a dataset of 215 rows and 15 columns but most of the values in the column 15 NaN so to handle those
missing values, we give a solution:

Solution:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('Datasets\\Placement_Dataset.csv')

mode = dataset["salary"].mode()[0]
median = dataset["salary"].median()
mean = dataset["salary"].mean()

x = (mean + median + mode)/3

dataset.fillna({"salary": x }, inplace = True)

print(dataset)

plt.plot(dataset.salary)
plt.show()

'''

# Random Data Distribution:

# NUMERICAL DATA:

'''

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

data = pd.read_csv("Datasets\\titanic.csv", usecols=["Survived" , "Age", "Fare"])
# print(data.isnull().mean() * 100)
print(data.head())

x = data.drop(["Survived"] , axis = 1)
y = data["Survived"]

trainx , testx , trainy , testy = train_test_split(x,y,test_size=0.3,random_state=30)

trainx["Age_imputed"] = trainx["Age"]
testx["Age_imputed"] = testx["Age"]

trainx.loc[trainx["Age_imputed"].isnull(), "Age_imputed"] = (
    trainx["Age"].dropna().sample(trainx["Age"].isnull().sum()).values
)

testx.loc[testx["Age_imputed"].isnull(), "Age_imputed"] = (
    trainx["Age"].dropna().sample(testx["Age"].isnull().sum()).values
)

sns.displot(data=trainx, x='Age', kind='kde', label='Original', color="blue")
sns.displot(data=trainx, x='Age_imputed', kind='kde', label='Imputed', color= "red")
plt.legend()
plt.show()

'''

# CATEGORICAL DATA:

'''

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

data = pd.read_csv("Datasets\\house-train.csv", usecols=["GarageQual","FireplaceQu", "SalePrice"])
# print(data.isnull().mean()*100)
# print(data.head())

x = data
y = data['SalePrice']

trainx, testx,trainy,testy = train_test_split(x,y,test_size=0.3,random_state=30)

trainx['GarageQual_imputed'] = trainx['GarageQual']
testx['GarageQual_imputed'] = testx['GarageQual']

trainx['FireplaceQu_imputed'] = trainx['FireplaceQu']
testx['FireplaceQu_imputed'] = testx['FireplaceQu']

missing_train_garage = trainx['GarageQual_imputed'].isnull()
trainx.loc[missing_train_garage, 'GarageQual_imputed'] = (
    trainx['GarageQual']
    .dropna()
    .sample(missing_train_garage.sum(), replace=True, random_state=42)
    .values
)

missing_test_garage = testx['GarageQual_imputed'].isnull()
testx.loc[missing_test_garage, 'GarageQual_imputed'] = (
    trainx['GarageQual'].dropna().sample(missing_test_garage.sum(), replace=True, random_state=42).values
)

# Fill FireplaceQu_imputed
missing_train_fire = trainx['FireplaceQu_imputed'].isnull()
trainx.loc[missing_train_fire, 'FireplaceQu_imputed'] = (
    trainx['FireplaceQu'].dropna().sample(missing_train_fire.sum(), replace=True, random_state=42).values
)

missing_test_fire = testx['FireplaceQu_imputed'].isnull()
testx.loc[missing_test_fire, 'FireplaceQu_imputed'] = (
    trainx['FireplaceQu'].dropna().sample(missing_test_fire.sum(), replace=True, random_state=42).values
)

import seaborn as sns
import matplotlib.pyplot as plt

# Plot for original FireplaceQu
plt.figure(figsize=(8,6))
for category in trainx['FireplaceQu'].dropna().unique():
    subset = trainx[trainx['FireplaceQu'] == category]
    sns.kdeplot(subset['SalePrice'], label=category, fill=False)
plt.legend(title='FireplaceQu')
plt.title('Sale Price Distribution by Original Fireplace Quality')
plt.xlabel('Sale Price')
plt.ylabel('Density')
plt.show()

# Plot for imputed FireplaceQu
plt.figure(figsize=(8,6))
for category in trainx['FireplaceQu_imputed'].dropna().unique():
    subset = trainx[trainx['FireplaceQu_imputed'] == category]
    sns.kdeplot(subset['SalePrice'], label=category, fill=False)
plt.legend(title='FireplaceQu_imputed')
plt.title('Sale Price Distribution by Imputed Fireplace Quality')
plt.xlabel('Sale Price')
plt.ylabel('Density')
plt.show()

'''

# Automatically selecting best imputation method:

'''

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# --- Load data ---
data = pd.read_csv("Datasets\\titanic.csv",
                   usecols=["Survived", "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"])

# --- Feature engineering ---
data["Family"] = data["SibSp"] + data["Parch"]
data.drop(columns=["SibSp", "Parch"], inplace=True)

x = data.drop(["Survived"], axis=1)
y = data["Survived"]
trainx, testx, trainy, testy = train_test_split(x, y, test_size=0.3, random_state=30)

# --- Create pipelines for Age and Fare separately so we can tune each imputer ---
age_pipe = Pipeline([
    ("imputer", SimpleImputer()),       # tune strategy via grid
    ("scaler", StandardScaler())
])

fare_pipe = Pipeline([
    ("imputer", SimpleImputer()),       # tune strategy via grid
    ("scaler", StandardScaler())
])

# categorical pipelines
sex_pipe = Pipeline([
    ("onehot", OneHotEncoder(sparse_output=False, handle_unknown="ignore"))
])

embark_pipe = Pipeline([
    ("ord", OrdinalEncoder(categories=[["Q", "S", "C"]], dtype=np.int32,
                           handle_unknown="use_encoded_value", unknown_value=-1))
])

# --- ColumnTransformer with named transformers ---
preprocessor = ColumnTransformer(
    transformers=[
        ("age", age_pipe, ["Age"]),
        ("fare", fare_pipe, ["Fare"]),
        ("sex", sex_pipe, ["Sex"]),
        ("embark", embark_pipe, ["Embarked"]),
        ("pass", "passthrough", ["Pclass", "Family"])
    ],
    remainder="drop"
)

# --- Full pipeline (named steps) ---
pipe = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=500))
])

# --- Grid search: tune imputer strategies for age and fare separately + classifier params ---
param_grid = {
    'preprocessor__age__imputer__strategy': ['mean', 'median'],
    'preprocessor__fare__imputer__strategy': ['mean', 'median'],
    'classifier__C': [0.1, 1.0, 10],
    'classifier__solver': ['lbfgs', 'liblinear']
}

grid_search = GridSearchCV(pipe, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(trainx, trainy)   # <- will run without the earlier error

print("Best params:", grid_search.best_params_)
best_model = grid_search.best_estimator_
predy = best_model.predict(testx)
print("Test accuracy:", accuracy_score(testy, predy))

'''

# KNN IMPUTATION:

'''

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import cross_val_score

df = pd.read_csv("Datasets\\titanic.csv" , usecols=["Survived","Age","SibSp","Parch","Sex","Embarked","Fare"])
df["Family"] = df["SibSp"] + df["Parch"]
df.drop(columns=["SibSp","Parch"] , axis = 1, inplace=True)

# print(df.head())

x = df.drop(columns=["Survived"] , axis=1)
y = df["Survived"]

# print(x.head())

trainx , testx, trainy, testy = train_test_split(x,y,test_size=0.3,random_state=30)

z = make_column_transformer(
    [KNNImputer(n_neighbors=5, weights="uniform"), ["Age"]],
    [KNNImputer(n_neighbors=5, weights="uniform"), ["Fare"]],
    remainder = "passthrough"
)

k = make_column_transformer(
    [OneHotEncoder(categories=[["male", "female"]], sparse_output=False, handle_unknown="ignore"), [2]],
    [OneHotEncoder(categories=[["Q","S","C"]], sparse_output=False, handle_unknown="ignore"), [3]],
    remainder="passthrough"
)

m = make_column_transformer(
    [MinMaxScaler(), [7]],
    remainder="passthrough"
)

l = LogisticRegression()

pipe = make_pipeline(z , k, m , l)

pipe.fit(trainx,trainy)
predy = pipe.predict(testx)

print("The accuracy score is: " +str(accuracy_score(testy,predy)))
print("The cross validation score is: " +str(np.mean(cross_val_score(pipe, trainx ,trainy , cv=10, scoring='accuracy'))))

'''

# ITERATIVE IMPUTER:

'''

Apparently, we dont need to specify the names of the column having missing values... the iterative imputer will automatically detect the missing
values and work on it. Also make sure to check the order of preprocessing:

OneHotEncoder/OrdinalEncoder → IterativeImputer → MinMaxScaler/StandardScaler/etc → LogisticRegression/etc

'''

# Example: 

'''

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline

df = pd.read_csv("Datasets\\titanic.csv", usecols=["Survived","Age","SibSp","Parch","Sex","Embarked","Fare"])
df["Family"] = df["SibSp"] + df["Parch"]
df.drop(columns=["SibSp","Parch"], inplace=True)

x = df.drop(columns=["Survived"])
y = df["Survived"]

# print(x.head())

trainx, testx, trainy, testy = train_test_split(x, y, test_size=0.3, random_state=30)

ls = LinearRegression()

e = make_column_transformer(
    [OneHotEncoder(categories=[["male", "female"]], sparse_output=False, handle_unknown="ignore"), ["Sex"]],
    [OneHotEncoder(categories=[["Q", "S", "C"]], sparse_output=False, handle_unknown="ignore"), ["Embarked"]],
    remainder="passthrough"
)

i = IterativeImputer(max_iter=10, imputation_order="ascending", estimator=ls)

s = make_column_transformer(
    [MinMaxScaler(), [7]],
    remainder="passthrough"
)

l = LogisticRegression()

pipe = make_pipeline(e, i, s, l)

pipe.fit(trainx, trainy)

predy = pipe.predict(testx)
print("Accuracy score:", accuracy_score(testy, predy))
print("Cross-validation score:", np.mean(cross_val_score(pipe, trainx, trainy, cv=10, scoring='accuracy')))

'''