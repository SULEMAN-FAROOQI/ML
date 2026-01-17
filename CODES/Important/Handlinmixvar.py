import numpy as np
import pandas as pd

df = pd.read_csv('Datasets\\titanic_mix_var.csv')
print(df.head())

# The dataset does not exsist in this device so,

# extract numerical part
df['number_numerical'] = pd.to_numeric(df["number"],errors='coerce',downcast='integer')
# extract categorical part
df['number_categorical'] = np.where(df['number_numerical'].isnull(),df['number'],np.nan)

df['cabin_num'] = df['Cabin'].str.extract('(/d)') # captures numerical part
df['cabin_cat'] = df['Cabin'].str[0] # captures the first letter

# extract the last bit of ticket as number
df['ticket_num'] = df['Ticket'].apply(lambda s: s.split()[-1])
df['ticket_num'] = pd.to_numeric(df['ticket_num'],
                                   errors='coerce',
                                   downcast='integer')

# extract the first part of ticket as category
df['ticket_cat'] = df['Ticket'].apply(lambda s: s.split()[0])
df['ticket_cat'] = np.where(df['ticket_cat'].str.isdigit(), np.nan,
                              df['ticket_cat'])

df.drop(["number","Ticket","Cabin"] , axis = 1, inplace = True)

print(df.head())