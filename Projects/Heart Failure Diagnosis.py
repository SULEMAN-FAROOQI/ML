import numpy as np
import pandas as pd
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.pipeline import make_pipeline
from lightgbm import LGBMClassifier
import warnings
warnings.filterwarnings('ignore')

data = pd.read_csv("Datasets\\heart_failure_clinical_records.csv")

# print(data.sample(10))
# print(data.shape)
# print(data.isna().sum())
# print(data.describe())
# print(data.duplicated().sum()

data = data.drop_duplicates()
# print(data.shape)

x = data.drop("DEATH_EVENT", axis = 1)
y = data["DEATH_EVENT"]

# print(x.sample(10))

trainx, testx, trainy, testy = train_test_split(x, y, test_size=0.3, random_state=33, stratify=y)

def ColumnTransformation(df):
    df = df.drop("time", axis = 1)
    return df

f = FunctionTransformer(ColumnTransformation)

m = LGBMClassifier(
    n_estimators=431,
    learning_rate=0.1866,
    num_leaves=99,
    max_depth=6,
    min_child_samples=3,
    subsample=0.9261,
    colsample_bytree=0.5327,
    reg_alpha=9.13e-08,
    reg_lambda=6.89e-06,
    random_state=42, 
    verbose = -1
)

pipe = make_pipeline(f,m)

pipe.fit(trainx,trainy)
proby = pipe.predict_proba(testx)[:, 1]
predy = pipe.predict(testx)

print("The ROC-AUC score is:", roc_auc_score(testy, proby))
print("")
print(classification_report(testy, predy))

sample = pd.DataFrame([{
    'age': 40.0,
    'anaemia': 0,
    'creatinine_phosphokinase': 90,
    'diabetes': 0,
    'ejection_fraction': 35,
    'high_blood_pressure': 0,
    'platelets': 255000.0,
    'serum_creatinine': 1.1,
    'serum_sodium': 136,
    'sex': 1,
    'smoking': 1,
    'time': 212
}])

prediction = pipe.predict(sample)
result = "chances of Survival." if prediction[0] == 0 else "very minor chances of survival"

print("The Patient having sample conditions have", result)