# This code represnets an example of a ML workflow. It predicts wheteher a person will survive the titanic accident based on their circumstances.

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split , cross_val_score
from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder , MinMaxScaler
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

dataframe = pd.read_csv("Datasets\\titanic.csv")
dataframe.drop(columns=["PassengerId","Name","Ticket","Cabin"], inplace=True)
print(dataframe.head())

trainx , testx , trainy , testy = train_test_split(dataframe.drop(columns=['Survived']), dataframe["Survived"] , test_size=0.3 , random_state=30)

x = make_column_transformer(
    (SimpleImputer(strategy="mean"), ["Age"]),
    (SimpleImputer(strategy="mean"), ["Fare"]),
    (OneHotEncoder(categories=[["male", "female"]], sparse_output=False, handle_unknown='ignore'), ["Sex"]),
    (OneHotEncoder(categories=[["S", "Q", "C"]], sparse_output=False, handle_unknown='ignore'), ["Embarked"]),
    (MinMaxScaler(), ["Pclass", "SibSp", "Parch"]),
    remainder="passthrough"
)

t = DecisionTreeClassifier(max_depth=5, min_samples_split=10, min_samples_leaf=5, random_state=30)

pipeline = make_pipeline(x,t)

pipeline.fit(trainx, trainy)
predy = pipeline.predict(testx)
print("The accuracy score is: " +str(accuracy_score(testy,predy)))
print("After cross validation our score is: " +str(np.mean(cross_val_score(pipeline, trainx ,trainy , cv=10, scoring='accuracy'))))

# Create a DataFrame instead of a NumPy array
new_data = pd.DataFrame([{
    "Pclass": 3,
    "Sex": "male",
    "Age": 34.5,
    "SibSp": 0,
    "Parch": 0,
    "Fare": 7.8292,
    "Embarked": "Q"
}])

prediction = pipeline.predict(new_data)

Survival_status = "Survived" if prediction[0] == 1 else "Died"
print("Prediction:", Survival_status)

# More Validation:

'''

test_cases = [
    {"Pclass": 1, "Sex": "female", "Age": 25, "SibSp": 0, "Parch": 0, "Fare": 100, "Embarked": "S"},  # → Should Survive
    {"Pclass": 3, "Sex": "male",   "Age": 40, "SibSp": 0, "Parch": 0, "Fare": 7.8, "Embarked": "Q"},  # → Should Die
    {"Pclass": 2, "Sex": "female", "Age": 10, "SibSp": 1, "Parch": 1, "Fare": 30,  "Embarked": "C"},  # → Should Survive
    {"Pclass": 3, "Sex": "male",   "Age": 60, "SibSp": 0, "Parch": 0, "Fare": 10,  "Embarked": "S"},  # → Should Die
]

for case in test_cases:
    pred = pipeline.predict(pd.DataFrame([case]))
    print(f"{case['Sex']}, Class {case['Pclass']}, Age {case['Age']} → {'Survived' if pred[0]==1 else 'Died'}")
    
'''