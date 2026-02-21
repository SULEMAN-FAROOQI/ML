import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX

df = pd.read_csv("Datasets\\Perrin Freres monthly champagne sales millions.csv")
df.dropna(inplace=True)

df["Month"] = pd.to_datetime(df["Month"])

# print(df.head())

# plt.plot(df["Month"],df["Sales"])
# plt.show()

# To check If the data is stationary, we find p-value. if p-value < 0.05 then data is stationary

'''

test_res = adfuller(df["Sales"])

labels = [
    "ADF Statistic",
    "p-value",
    "Lags Used",
    "Number of Observations"
]

for value, label in zip(test_res[:4], labels):
    print(label, ":", round(value, 4))

'''

# Critical Values:

'''

The 5th element of adfuller() output is a dictionary of critical values. critical values help you decide whether to reject 
a statistical hypothesis — typically in tests like the Augmented Dickey Fuller (ADF) test. The test includes:

1. Null hypothesis (H₀): The series has a unit root so Not stationary
2. Alternative hypothesis (H₁): The series is stationary

A critical value is a cutoff number used to decide whether your ADF Statistic is extreme enough to reject the null hypothesis.

If ADF Statistic < critical value (more negative), Reject H₀ → The data is stationary.
If ADF Statistic > critical value, Fail to reject H₀ → The data is not stationary.

'''

# p-value : 0.3639 so data is not stationary.

# Differencing: Converting data to stationary:

df["Seasonal first difference"] = df["Sales"] - df["Sales"].shift(12) 
# The dataset is of two years of sales so it will have seasonal shifts of 12 months.

'''

plt.plot(df["Month"],df["Seasonal first difference"])
plt.show()

Seasonal_test_res = adfuller(df["Seasonal first difference"].dropna())

labels = [
    "ADF Statistic",
    "p-value",
    "Lags Used",
    "Number of Observations"
]

for value, label in zip(Seasonal_test_res[:4], labels):
    print(label, ":", round(value, 16))

'''

# p-value : 2.06058e-11 which is less than 0.05 so now the data is converted to startionary.

# Determining the values of p , d and q:

'''

fig = plt.figure(figsize=(12, 8))

ax1 = fig.add_subplot(211)
fig = plot_acf(
    df['Seasonal first difference'].dropna(), 
    lags=40, 
    ax=ax1
)

ax2 = fig.add_subplot(212)
fig = plot_pacf(
    df['Seasonal first difference'].dropna(), 
    lags=10, 
    ax=ax2
)

plt.show()

'''

# The value of p can be specified from pacf plot. 
# The value of q can be best specified from acf plot.
# We only shifted our data 1 time so d = 1

arm = ARIMA(df["Sales"], order=(1,1,0)) # order = (p,d,q)
results1 = arm.fit()

# Now plot the forecast against the real data
df['forecast1'] = results1.predict(start=70, end=103, dynamic=True)
plt.figure(figsize=(12, 8))
plt.plot(df["Month"], df["Sales"], label="Actual Sales")
plt.plot(df["Month"], df["forecast1"], label="Forecast", color='red')
plt.legend()

# As we can see the arima model is not very good on seasonal data so now we will use sarima model.

# P=1, D=1, Q=1, s=12 (Seasonal cycle)
sarm = SARIMAX(df['Sales'], 
                order=(1, 1, 0), 
                seasonal_order=(1, 1, 0, 12))

results2 = sarm.fit()

# Now plot the forecast against the real data
df['forecast2'] = results2.predict(start=70, end=103, dynamic=True)

plt.figure(figsize=(12, 8))
plt.plot(df["Month"], df["Sales"], label="Actual Sales")
plt.plot(df["Month"], df["forecast2"], label="Forecast", color='red')
plt.legend()

plt.show()
