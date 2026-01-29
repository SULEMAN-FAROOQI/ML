# Handling Unbalanced data:

'''

It is a dataset with an unequal class(type given as a column) distribution. For example an imbalanced class of 1500
rows and 30 columns. In which 1250 belong to one group or category and 250 belongs to another group or category.
The two important functions used are:

1. variable.sample(): It is used to generate a specifed sample of unique elements from a given sequence like a dataset.
Argument 'n' is used to define the length of the new datatype. 

2. pandas.concat(): This function is the part of the pandas library. It is used to concatenate pandas objects (like DataFrames or Series) 
along a particular axis.

'''

import numpy as np
import pandas as pd

dataframe = pd.read_csv("\\Users\\dell\\Desktop\\DEVELOPMENT\\DOCUMENTS\\S.PY\\ML\\Datasets\\credit_data.csv.crdownload")
# print(dataframe)

# print(dataframe['Class'].value_counts())

legit = dataframe[dataframe.Class == 0]
fraud = dataframe[dataframe.Class == 1]

# print(legit.shape)
# print(fraud.shape)

x =  legit.sample(n = 150)

new_dataframe = pd.concat([x, fraud], axis = 0)
print(new_dataframe.head())
print(new_dataframe.tail())
print(new_dataframe.shape)
