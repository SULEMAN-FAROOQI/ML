# This code represnets an example of a ML workflow. It predicts wheteher a submarine will encounter mine or a rock based on certain parameters and values.

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split , cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv("C:\\Users\\sulem\\Desktop\\Codes\\ML\\Datasets\\sonar data.csv", header = None)
df = df.rename(columns={60: "Target"})
new_col = [f"x{i+1}" for i in range(60)]
new_col.append("Target")
df.columns = new_col

# print(df.head())
# print(df.isna().sum())
# print(df["Target"].value_counts())

x = df.drop("Target", axis = 1)
y = df["Target"]

encoder = LabelEncoder()
y = encoder.fit_transform(y)

columnx = x.columns.tolist()

trainx , testx , trainy, testy = train_test_split(x, y, test_size=0.3, stratify=y, random_state=33)

z = make_column_transformer(
    (StandardScaler(), columnx),
    remainder="passthrough"
)

log = SVC(
    kernel = 'rbf',     
    C      = 100,      
    gamma  = 'scale',  
    shrinking = True,   
    tol    = 0.0001    
)

pipe = make_pipeline(z,log)

pipe.fit(trainx, trainy)
predy = pipe.predict(testx)
print("The accuracy score is: " +str(accuracy_score(testy,predy)))
print("After cross validation our score is: " +str(np.mean(cross_val_score(pipe, trainx ,trainy , cv=10, scoring='accuracy'))))

data = [
    0.0200, 0.0371, 0.0428, 0.0207, 0.0954, 0.0986, 0.1539, 0.1601, 0.3109, 0.2111,
    0.1609, 0.1582, 0.2238, 0.0645, 0.0660, 0.2273, 0.3100, 0.2999, 0.5078, 0.4797,
    0.5783, 0.5071, 0.4328, 0.5550, 0.6711, 0.6415, 0.7104, 0.8080, 0.6791, 0.3857,
    0.1307, 0.2604, 0.5121, 0.7547, 0.8537, 0.8507, 0.6692, 0.6097, 0.4943, 0.2744,
    0.0510, 0.2834, 0.2825, 0.4256, 0.2641, 0.1386, 0.1051, 0.1343, 0.0383, 0.0324,
    0.0232, 0.0027, 0.0065, 0.0159, 0.0072, 0.0167, 0.0180, 0.0084, 0.0090, 0.0032
]

column_names = [f"x{i+1}" for i in range(60)]

new_data = pd.DataFrame([dict(zip(column_names, data))])

prediction = pipe.predict(new_data)

result = "Mine" if prediction[0] == 0 else "Rock"
print("The Submarine encountered a:", result)
