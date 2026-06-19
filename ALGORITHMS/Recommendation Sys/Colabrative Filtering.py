import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

books = pd.read_csv("Datasets\\BookRecomData\\Books.csv", low_memory=False)
ratings = pd.read_csv("Datasets\\BookRecomData\\Ratings.csv", low_memory=False)
users = pd.read_csv("Datasets\\BookRecomData\\Users.csv", low_memory=False)

print(books.isnull().sum())
print("-")
print(ratings.isnull().sum())
print("-")
print(users.isnull().sum())
