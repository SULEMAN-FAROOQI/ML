import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import Prophet
from sklearn.metrics import r2_score

data = pd.read_csv("Datasets\\PJME_hourly.csv")

pal = sns.color_palette()
data.plot(style=".", figsize=(10,5), color = pal[0], title="PJME")