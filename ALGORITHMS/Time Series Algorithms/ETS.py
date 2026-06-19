import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.exponential_smoothing.ets import ETSModel

df = pd.read_csv("Datasets\\Perrin Freres monthly champagne sales millions.csv")
df.dropna(inplace=True)

df["Month"] = pd.to_datetime(df["Month"])

# Unlike Arima or Sarima it does not require data to be stationary. ETS handles the raw trend and seasonality internally.

ets = ETSModel(
    df['Sales'], 
    error="mul", 
    trend="add", 
    seasonal="mul", 
    seasonal_periods=12 # A year has 12 months
)

# 3. Fit the model
model = ets.fit()

# 4. Forecast 2 years (24 months) into the future
forecast = model.get_prediction(start=len(df), end=len(df)+23)
forecast_df = forecast.summary_frame(alpha=0.05)

plt.figure(figsize=(15, 6)) 

plt.subplot(1, 2, 1) # 1 row, 2 columns, index 1
plt.plot(df.index, df["Sales"]) # Using index if it's a DateTimeIndex
plt.title("Original Sales Data")
plt.xlabel("Date")
plt.ylabel("Sales")

plt.subplot(1, 2, 2) # 1 row, 2 columns, index 2
plt.plot(df['Sales'], label='Actual Sales')
plt.plot(forecast_df['mean'], label='ETS Forecast', color='red', linestyle='--')
plt.fill_between(forecast_df.index, forecast_df['pi_lower'], forecast_df['pi_upper'], color='pink', alpha=0.3)
plt.title('Champagne Sales Forecast')
plt.legend()

plt.tight_layout()
plt.show()
