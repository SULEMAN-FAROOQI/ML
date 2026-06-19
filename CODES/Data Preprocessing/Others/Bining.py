import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.preprocessing import Binarizer
from sklearn.compose import make_column_transformer

tree = DecisionTreeClassifier()

data = pd.read_csv("Datasets\\titanic.csv", usecols=["Age","Fare","SibSp","Parch","Survived"])

mode = data["Age"].mode()[0]
median = data["Age"].median()
mean = data["Age"].mean()
z = (mean + median + mode)/3
data.fillna({"Age" : z }, inplace=True)

data["Family"] = data["SibSp"] + data["Parch"]
data.drop(["SibSp" , "Parch"], axis=1, inplace= True)
data.dropna(inplace=True)
# print(data.head())

x = data.drop("Survived", axis=1)
# print(x.head())
y = data["Survived"]
# print(y.head())

trainx, testx, trainy, testy = train_test_split(x , y , test_size=0.3, random_state=30)

trf = make_column_transformer(
    [KBinsDiscretizer(n_bins=5 , encode="ordinal", strategy="quantile", quantile_method='averaged_inverted_cdf'), ["Age"]],
    [KBinsDiscretizer(n_bins=5 , encode="ordinal", strategy="quantile" , quantile_method='averaged_inverted_cdf'), ["Fare"]],
    [Binarizer(copy=False) , ["Family"]],
    remainder = "passthrough"
)

Xtrain = trf.fit_transform(trainx)
Xtest = trf.transform(testx)

tree.fit(Xtrain, trainy)
predy = tree.predict(Xtest)
print("The Accuracy score is: " +str(accuracy_score(testy, predy)))
print("After cross validation our score is: " +str(np.mean(cross_val_score(tree, Xtrain ,trainy , cv=10, scoring='accuracy'))))