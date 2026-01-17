# Train Test Split Functionality:

'''

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_linnerud

dataset = load_linnerud()
# print(dataset)

x = pd.DataFrame(data=dataset.data, columns=dataset.feature_names)
y = pd.DataFrame(data=dataset.target, columns=dataset.target_names)

# print(x.head)
# print(y.head)

scaler = StandardScaler()

x_std = scaler.fit_transform(x)
y_std = scaler.fit_transform(y)

trainx, testx, trainy, testy = train_test_split(x,y, test_size=0.3, random_state=7)

print(x_std.shape)
print(trainx.shape)
print(testx.shape)

print(y_std.shape)
print(trainy.shape)
print(testy.shape)

'''