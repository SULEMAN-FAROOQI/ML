# This code represnets an example of a ML workflow. It predicts wheteher a female has diabetes or not based on certain parameters like blood and glucose level.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler , FunctionTransformer
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.model_selection import train_test_split , cross_val_score 
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from sklearn.impute import KNNImputer
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv("Datasets\\diabetes.csv")
df.rename(columns={'Outcome': 'Diagnosis'}, inplace=True)

# print(df.head())
# print(df.shape)
# print(df.describe())

x = df.drop("Diagnosis", axis = 1)
y = df["Diagnosis"]

def ColumnTransformation(df):
    df = df.copy() 
    df[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] = df[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']].replace(0, np.nan)
    df['BMI_Glucose'] = df['BMI'] * df['Glucose']
    df['Age_Pregnancies'] = df['Age'] * df['Pregnancies']
    return df

f = FunctionTransformer(ColumnTransformation)

# Plotting each column to check linearity

'''

for i in range(x.shape[1] - 1):
    plt.figure()
    plt.scatter(x.iloc[:,i], x.iloc[: , i + 1] , c= y)
    plt.xlabel(x.columns[i])
    plt.ylabel(x.columns[i + 1])
    plt.show()

'''
# By the Figures we deduce that the data is non linear so,

# print(y.value_counts())

trainx , testx , trainy , testy = train_test_split(x,y , stratify=y, random_state=33, test_size=0.3)

z = make_column_transformer(
    (make_pipeline(
        KNNImputer(n_neighbors=5), # Missing values in medical data tend to cluster (e.g. obese patients have similar glucose/insulin patterns), which KNN captures well.
        StandardScaler()
    ), ColumnTransformation(x).columns.tolist()),
    remainder="passthrough"
)

m = XGBClassifier(
    n_estimators      = 207,
    max_depth         = 3,
    learning_rate     = 0.0181,
    subsample         = 0.817,
    colsample_bytree  = 0.842,
    min_child_weight  = 6,
    reg_alpha         = 0.104,
    use_label_encoder = False,
    eval_metric       = 'logloss',
    random_state      = 33
)

pipeline = make_pipeline(f,z,m)

pipeline.fit(trainx,trainy)

probay = pipeline.predict_proba(testx)[:, 1]
print("The ROC-AUC score is:",roc_auc_score(testy, probay))
print("After cross validation our score is: " +str(np.mean(cross_val_score(pipeline, trainx ,trainy , cv=10, scoring='roc_auc'))))

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

result = "Negative" if prediction[0] == 0 else "Positive"
print("Prediction:", result)
