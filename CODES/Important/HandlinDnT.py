import numpy as np
import pandas as pd

date = pd.read_csv('Datasets\\orders.csv')
time = pd.read_csv('Datasets\\messages.csv')

print(date.head())
print(time.head())

# The datatype of our date and time columns in above is an object or string first we will convert it into date and time.

# Date Manipulation:

# Converting to date datatype:
date['date'] = pd.to_datetime(date['date'])

date['date_year'] = date['date'].dt.year # Extract year
date['date_month_no'] = date['date'].dt.month # Extract Month
date['date_month_name'] = date['date'].dt.month_name()
date['date_day'] = date['date'].dt.day
date['date_dow_name'] = date['date'].dt.day_name()
date['date_week'] = date['date'].dt.week
date['semester'] = np.where(date['quarter'].isin([1,2]), 1, 2)

# Time Manipulation:

# Converting to datetime datatype:
time['date'] = pd.to_datetime(time['date'])

time['hour'] = time['date'].dt.hour
time['min'] = time['date'].dt.minute
time['sec'] = time['date'].dt.second
time['time'] = time['date'].dt.time

print(time.head())