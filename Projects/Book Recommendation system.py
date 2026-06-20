# This code represnets an example of a ML workflow. It recommends different books on their similarities with other books and user ratings.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_colwidth', 30)

books = pd.read_csv(r"C:\\Users\sulem\\OneDrive\\Desktop\\Codes\\ML\\Datasets\\BookRecomData\\Books.csv", low_memory=False)
books = books.drop(columns=['Image-URL-S', 'Image-URL-L'])
ratings = pd.read_csv(r"C:\\Users\sulem\\OneDrive\\Desktop\\Codes\\ML\\Datasets\\BookRecomData\\Ratings.csv", low_memory=False)
users = pd.read_csv(r"C:\\Users\sulem\\OneDrive\\Desktop\\Codes\\ML\\Datasets\\BookRecomData\\Users.csv", low_memory=False)

'''

books.isnull().sum()
print("-")
ratings.isnull().sum()
print("-")
users.isnull().sum()

'''

Books_with_Ratings = ratings.merge(books, on="ISBN")
Num_Rate = Books_with_Ratings.groupby("Book-Title")["Book-Rating"].count().reset_index()
Num_Rate.rename(columns={"Book-Rating": "Num of Ratings"}, inplace=True)

Avg_Num_Rate = Books_with_Ratings.groupby("Book-Title")["Book-Rating"].mean().reset_index()
Avg_Num_Rate.rename(columns={"Book-Rating": "Avg Rating"}, inplace=True)

'''

Books_with_Ratings.head()
print("-")
Num_Rate.head()
print("-")
Avg_Num_Rate.head()

'''

Popularity_Matrix = Num_Rate.merge(Avg_Num_Rate, on="Book-Title")
Popularity_Matrix = Popularity_Matrix[Popularity_Matrix["Num of Ratings"] >= 250].sort_values("Avg Rating", ascending=False)
Popularity_Matrix = Popularity_Matrix.merge(books, on="Book-Title").drop_duplicates("Book-Title")[
    ["Book-Title", "Book-Author", "Year-Of-Publication", "Publisher", "Num of Ratings", "Avg Rating", "Image-URL-M"]
]

'''

Popularity_Matrix.head(50)
Popularity_Matrix.shape

'''

u = Books_with_Ratings.groupby("User-ID").count()["Book-Rating"] > 200
u = u[u].index

Filtered_Ratings = Books_with_Ratings[Books_with_Ratings["User-ID"].isin(u)]

'''

Filtered_Ratings.head()
Filtered_Ratings.shape

'''

k = Filtered_Ratings.groupby("Book-Title").count()["Book-Rating"] >= 50
Famous_Books = k[k].index
Final_Ratings = Filtered_Ratings[Filtered_Ratings["Book-Title"].isin(Famous_Books)]

Pt = Final_Ratings.pivot_table(index="Book-Title", columns="User-ID", values="Book-Rating")
Pt.fillna(0,inplace=True)
# Pt.head(10)

Similarity_Score = cosine_similarity(Pt)
Similarity_Score

def Recommend(book_name):
    index = np.where(Pt.index == book_name)[0][0]
    similar_items = sorted(list(enumerate(Similarity_Score[index])), key=lambda x: x[1], reverse=True)[1:11]
    for i in similar_items:
        print(Pt.index[i[0]])

Recommend("Harry Potter and the Chamber of Secrets (Book 2)")
