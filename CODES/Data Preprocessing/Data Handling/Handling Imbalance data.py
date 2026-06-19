import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

df = pd.read_csv("Datasets\\imbalance.csv")

us = RandomUnderSampler(random_state=33)
os = RandomOverSampler(random_state=33)
smote = SMOTE(k_neighbors=3, random_state=33)

logrg = LogisticRegression(solver="saga", max_iter=10000)

x = df.drop("placement_status", axis = 1)
y = df["placement_status"]

x0 , y0 = us.fit_resample(x,y)
x1 , y1 = os.fit_resample(x,y)
x2 , y2 = smote.fit_resample(x,y)

trainx0 , testx0 , trainy0 , testy0 = train_test_split(x0,y0, test_size=0.3, random_state=33)
trainx1 , testx1 , trainy1 , testy1 = train_test_split(x1,y1, test_size=0.3, random_state=33)
trainx2 , testx2 , trainy2 , testy2 = train_test_split(x2,y2, test_size=0.3, random_state=33)

logrg.fit(trainx0, trainy0)
predy0 = logrg.predict(testx0)
print("The Accuracy score after Under Sampling is:","\n",classification_report(testy0,predy0))

logrg.fit(trainx1, trainy1)
predy1 = logrg.predict(testx1)
print("The Accuracy score after Over Sampling is:","\n",classification_report(testy1,predy1))

logrg.fit(trainx2, trainy2)
predy2 = logrg.predict(testx2)
print("The Accuracy score after using Synthetic Minority Oversampling technique is:","\n",classification_report(testy2,predy2))

