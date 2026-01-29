# A scikit-learn pipeline is a sequence of steps that process your features (x) — not your target (y) — 
# up until the final estimator (like LinearRegression or LogisticRegression).

# pipeline = make_pipeline(preprocessor, model)

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import SelectKBest,chi2
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

dataframe = pd.read_csv("Datasets\\titanic.csv")
dataframe.drop(columns=["PassengerId","Name","Ticket","Cabin"], inplace=True)
# print(dataframe.head())

trainx , testx , trainy , testy = train_test_split(dataframe.drop(columns=['Survived']), dataframe["Survived"] , test_size=0.3 , random_state=30)

x = make_column_transformer(
    (SimpleImputer(strategy="most_frequent"), ["Age"]),
    (SimpleImputer(strategy="mean"), ["Fare"]),
    (OneHotEncoder(categories=[["male", "female"]], sparse_output=False, handle_unknown='ignore'), ["Sex"]),
    (OneHotEncoder(categories=[["S", "Q", "C"]], sparse_output=False, handle_unknown='ignore'), ["Embarked"]),
    (MinMaxScaler(), ["Pclass", "SibSp", "Parch"]),
    remainder="passthrough"
)

k = SelectKBest(score_func=chi2, k = 9)
t = DecisionTreeClassifier()

pipeline = make_pipeline(x , k , t)

pipeline.fit(trainx, trainy)
predy = pipeline.predict(testx)
print("The accuracy score is: " +str(accuracy_score(testy,predy)))
print("After cross validation our score is: " +str(np.mean(cross_val_score(pipeline, trainx ,trainy , cv=10, scoring='accuracy'))))

'''

The transformed dataframe (after dropping 'PassengerId', 'Name', 'Ticket', 'Fare', 'Cabin') has the following columns and indices:

IndexColumn           NameData                   Type
0                     Survived                   int64 
1                     Pclass                     int64
2                     Sex                        object 
3                     Age                        float64
4                     SibSp                      int64
5                     Parch                      int64
6                     Embarked                   object

The actual columns and indices of trainx when it first enters transformer x are:

IndexColumn           Name
0                     Pclass
1                     Sex
2                     Age
3                     SibSp
4                     Parch
5                     Embarked

By setting remainder="passthrough", the transformer x passes all other columns, including 'Sex' (index 1) and 'Embarked' (index 5), 
through to the next step without change. However, the SimpleImputer is designed to process the selected columns first, and then the 
remainder columns are appended. When x completes, the output structure is:

Imputed Column: Age (Numerical, index 2 in original)
Passthrough Columns: Pclass, Sex, SibSp, Parch, Embarked (Indices 0, 1, 3, 4, 5 in original)

The new column will be:

IndexColumn          Name                       Data_Type
0                    Age(Imputed)               Numerical
1                    Pclass                     Numerical
2                    Sex                        String ('male', 'female')
3                    SibSp                      Numerical
4                    Parch                      Numerical
5                    Embarked                   String ('S', 'C', 'Q')

The total number of columns entering z is 2 + 3 + 4 = 9 columns. The indices will be from 0 to 8.

Also categories must be a list of list nd we should only the name of the column in transfer, if we give it like this df["name of col"]
we will be giving the series not the column.

The column fare was added later.

'''