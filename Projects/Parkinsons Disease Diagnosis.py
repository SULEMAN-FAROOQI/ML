import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, precision_score
from lightgbm import LGBMClassifier
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv("Datasets\\parkinsons.csv")

x = df.drop(["status"], axis=1) 
y = df["status"]

trainx, testx, trainy, testy = train_test_split(x, y, test_size=0.3, random_state=33, stratify=y)

def ColumnSelection(z):
    z = z.drop("name", axis=1)         
    return z

pipe = make_pipeline(
    FunctionTransformer(ColumnSelection),
    StandardScaler(),
    PCA(n_components=0.90),
    LGBMClassifier(n_estimators=403, max_depth=4, learning_rate=0.194,
                   num_leaves=16, min_child_samples=41, subsample=0.877,
                   colsample_bytree=0.564, reg_alpha=0.00156, reg_lambda=0.000877,
                   scale_pos_weight=0.327, verbosity=-1)
)

pipe.fit(trainx, trainy)
predy = pipe.predict(testx)

print("Accuracy :", accuracy_score(testy, predy))
print("Precision:", precision_score(testy, predy))

print("--------------------------------------------------------------")

sample = pd.DataFrame({
    'name': ['phon_R01_S13_3'],
    'MDVP:Fo(Hz)': [124.445],
    'MDVP:Fhi(Hz)': [135.069],
    'MDVP:Flo(Hz)': [117.495],
    'MDVP:Jitter(%)': [0.00431],
    'MDVP:Jitter(Abs)': [0.00003],
    'MDVP:RAP': [0.00141],
    'MDVP:PPQ': [0.00167],
    'Jitter:DDP': [0.00422],
    'MDVP:Shimmer': [0.02184],
    'MDVP:Shimmer(dB)': [0.197],
    'Shimmer:APQ3': [0.01241],
    'Shimmer:APQ5': [0.01024],
    'MDVP:APQ': [0.01685],
    'Shimmer:DDA': [0.03724],
    'NHR': [0.00479],
    'HNR': [25.135],
    'RPDE': [0.553134],
    'DFA': [0.775933],
    'spread1': [-6.650471],
    'spread2': [0.254498],
    'D2': [1.840198],
    'PPE': [0.103561]
})

predictions = pipe.predict(sample)
result = "has Parkinsons Disease." if predictions[0] == 1 else "does not have Parkinsons Disease."
print("Patient",sample["name"].values[0],result)
