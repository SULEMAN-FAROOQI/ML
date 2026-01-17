import numpy as np
import pandas as pd

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

import seaborn as sns

df = pd.read_csv('train.csv')
df.head()

df['Title'] = df['Name'].str.split(', ', expand=True)[1].str.split('.', expand=True)[0]

# It is splititing the strings by ", "... the [1] indicates the string after comma, then it is splitting the 
# strings by "." ... the [0] indicates the string before dot.