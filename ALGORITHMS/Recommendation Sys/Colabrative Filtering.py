import numpy as np
import pandas as pd

books = pd.read_csv("Datasets\\BookRecomData\\Books.csv")
ratings = pd.read_csv("Datasets\\BookRecomData\\Ratings.csv")
users = pd.read_csv("Datasets\\BookRecomData\\Users.csv")

# Merge the datasets