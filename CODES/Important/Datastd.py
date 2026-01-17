'''

In Data Standardization, we transform the data to have a mean of 0 and a standard deviation of 1. First we split the data into Training and 
Testing features and Training and Testing Targets. Then we fit the StandardScaler on the Training features and transform both the Training and 
Testing features.

1. scaler.fit(variable) = This function computes the mean and standard deviation for each feature in the dataset and stores them.
2. scaler.transform(variable) = It transforms the data by subtracting the mean and dividing by the standard deviation for each feature.
3. scaler.fit_transform(variable) = It is the combination of fit and transform function.

'''

# Example (Breast Cancer Dataset Standardization):

'''

import pandas as pd
from sklearn.preprocessing import StandardScaler 
from sklearn.datasets import load_breast_cancer
import numpy as np

dataset = load_breast_cancer()

x = pd.DataFrame(data = dataset.data, columns=dataset.feature_names)
y = dataset.target

print(dataset.data.std())

# print(x.head())
# print(y)

scaler = StandardScaler()

x_std = scaler.fit_transform(x)
datax = pd.Dataframe(data = x_std, columns = dataset.feature_names)
print(x_std.std())

'''

# Example (Diabetes Dataset Standardization):

'''

from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.datasets import load_diabetes

scaler = StandardScaler()

dataset = load_diabetes()
# print(dataset)

x = pd.DataFrame(data = dataset.data, columns=dataset.feature_names)
y = dataset.target

print(dataset.data.std())

x_std = scaler.fit_transform(x)
datax = pd.Dataframe(data = x_std, columns = dataset.feature_names)
print(x_std.std())

'''

# Example (Digit Dataset Standardization):

'''

from sklearn.preprocessing import StandardScaler
import pandas as pd
import sklearn.datasets

scaler = StandardScaler()

dataset = sklearn.datasets.load_digits()

x = pd.DataFrame(data= dataset.data, columns=dataset.feature_names)
y = dataset.target

print(dataset.data.std())

x_std = scaler.fit_transform(x)
datax = pd.Dataframe(data = x_std, columns = dataset.feature_names)
print(x_std.std())

'''
