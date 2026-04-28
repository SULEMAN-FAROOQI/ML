# # This code represnets an example of a ML workflow. It predicts wheteher a female has diabetes or not based on certain parameters like blood and glucose level.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.model_selection import train_test_split , cross_val_score , GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv("C:\\Users\\sulem\\Desktop\\Codes\\ML\\Datasets\\diabetes.csv")
df.rename(columns={'Outcome': 'Diagnosis'}, inplace=True)

# print(df.head())
# print(df.shape)
# print(df.describe())

x = df.drop("Diagnosis", axis = 1)
y = df["Diagnosis"]

columnx = x.columns.tolist()

# Plotting each column to check linearity

'''

for i in range(x.shape[1] - 1):
    plt.figure()
    plt.scatter(x.iloc[:,i], x.iloc[: , i + 1] , c= y)
    plt.xlabel(x.columns[i])
    plt.ylabel(x.columns[i + 1])
    plt.show()

'''

# print(y.value_counts())

trainx , testx , trainy , testy = train_test_split(x,y , stratify=y, random_state=33, test_size=0.3)

z = make_column_transformer(
    [StandardScaler() , columnx], 
    remainder="passthrough"
)

# By the Figures we deduce that the data is non linear so,

'''

param = {
    'C': [0.1, 1, 10],
    'penalty': ['l1', 'l2'],
    'solver': ['sag','saga', 'liblinear'], # Supports both l1 and l2
    'max_iter': [1000]       # Ensures it converges
}

grid = GridSearchCV(
    LogisticRegression(),
    param,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

grid.fit(trainx, trainy)

print("Best Params:", grid.best_params_)
print("CV Accuracy:", grid.best_score_)

'''

m = LogisticRegression(C =  1, max_iter = 1000, penalty = 'l1', solver = 'liblinear')

pipeline = make_pipeline(z,m)

pipeline.fit(trainx,trainy)
predy = pipeline.predict(testx)
print("The accuracy score is: " +str(accuracy_score(testy,predy)))
print("After cross validation our score is: " +str(np.mean(cross_val_score(pipeline, trainx ,trainy , cv=10, scoring='accuracy'))))

new_data = pd.DataFrame([{
    "Pregnancies": 6,
    "Glucose": 148,
    "BloodPressure": 72,
    "SkinThickness": 35,
    "Insulin": 0,
    "BMI": 33.6,
    "DiabetesPedigreeFunction": 0.627,
    "Age": 50
}])

prediction = pipeline.predict(new_data)

result = "Negative" if prediction[0] == '0' else "Positive"
print("Prediction:", result)
