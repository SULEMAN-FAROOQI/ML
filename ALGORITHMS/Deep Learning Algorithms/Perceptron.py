import numpy as np
import pandas as pd
from sklearn.linear_model import Perceptron
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score

df = pd.read_csv("Datasets\\job.csv")

x = df.drop(columns = ["Placement_Status", "Secondary_Percentage"], axis = 1)
y = df["Placement_Status"]

label = LabelEncoder()
scaler = StandardScaler()

x = scaler.fit_transform(x)
y = label.fit_transform(y)

trainx, testx, trainy, testy = train_test_split(x, y, test_size=0.3, random_state=33, stratify=y)

'''

sc = plt.scatter(x[:, 0], x[:, 1], c=y)
plt.colorbar(sc, label="Placement_Status")
plt.show()

'''

p = Perceptron()

p.fit(trainx, trainy)
predy = p.predict(testx)

print("The Accuracy Score is:",accuracy_score(testy,predy))
print("The Precision Score is:",precision_score(testy,predy))

'''

print(p.coef_) # Values of weights
print(p.intercept_) # Value of bias or slope

'''

plot_decision_regions(x, y, clf = p, legend = 2)
plt.show()