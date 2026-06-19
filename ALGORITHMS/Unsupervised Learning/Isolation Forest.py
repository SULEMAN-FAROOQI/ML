import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import warnings

warnings.filterwarnings("ignore")

scaler = StandardScaler()

data = pd.read_csv("Datasets\\anamoly.csv")

x = data.drop("fraud", axis=1)
y = data["fraud"]

x = scaler.fit_transform(x)

trainx ,testx, trainy ,testy = train_test_split(x,y, test_size=0.3, random_state=33)

ifr = IsolationForest(n_estimators=300)

ifr.fit(trainx) # Unsupervised trainy will not be added
predy_ifr = ifr.predict(testx)

# Correct mapping: -1 → 1 (fraud), 1 → 0 (normal)

predy_ifr = np.where(predy_ifr == -1, 1, 0)

print("The Accuracy score after using Isolation forest for anamoly detection is:",accuracy_score(testy,predy_ifr))

# pip install eif remember search when it can support python 3.13

# 1. Define your new data as a 2D array (Isolation Forest expects a matrix)
new_data = np.array([[146.7954605564706, 6, 1713, 368, 9.8861758621572413, 8, 78]])

# 2. Scale the data using the ALREADY fitted scaler
# Crucial: Use .transform(), NOT .fit_transform()
new_data_scaled = scaler.transform(new_data)

# 3. Predict
prediction = ifr.predict(new_data_scaled)

# 4. Map the result (just like you did for the test set)
is_fraud = "Fraud (Anomaly)" if prediction[0] == -1 else "Normal"

print("Prediction:",is_fraud)
