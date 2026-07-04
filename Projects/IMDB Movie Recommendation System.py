import numpy as np
import pandas as pd
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity # Similarity is 0 (No Similarity) and 1 (Similar)

pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', 100)

movies = pd.read_csv("C:\\Users\\sulem\\OneDrive\\Desktop\\Codes\\ML\\Datasets\\imdb_movies.csv")

# print(movies.head())
# print(movies.shape)

movies = movies.drop(columns = ["date_x", "score", "orig_title", "status", "orig_lang", "budget_x", "revenue", "country"])
movies.drop_duplicates(inplace = True)
movies.reset_index(drop=True, inplace=True)

# print(movies.head())
# print(movies.isnull().sum())
# print(movies.dropna(inplace = True))
# print(movies.duplicated().sum())
# print(movies.shape)

movies["genre"] = movies["genre"].apply(lambda x: [i.strip() for i in x.split(",")] if isinstance(x,str) else [])

movies["cast"] = movies["crew"].apply(lambda x: [" ".join(i.strip().split()[:2]) for i in x.split(",")] if isinstance(x, str) else [])
movies["cast"] = movies["cast"].apply(lambda x:x[:5])

movies["overview"] = movies["overview"].apply(lambda x : x.split())

movies = movies.drop("crew", axis = 1)

movies["cast"] = movies["cast"].apply(lambda x : [i.replace(" ","") for i in x])
movies["genre"] = movies["genre"].apply(lambda x : [i.replace(" ","") for i in x])

movies["Tags"] = movies["genre"] + movies["cast"] + movies["overview"]
movies = movies.drop(columns = ["genre", "cast", "overview"])

movies.insert(loc = 0, column = "Movies_ID", value = np.arange(1, len(movies) + 1))

movies["Tags"] = movies["Tags"].apply(lambda x : " ".join(x))
movies["Tags"] = movies["Tags"].apply(lambda x : x.lower())

# print(movies.head())

ps = PorterStemmer()

def stem(text):
    L = []
    for i in text.split():
        L.append(ps.stem(i))

    return " ".join(L)

movies["Tags"] = movies["Tags"].apply(stem)

cv = TfidfVectorizer(max_features=9000, stop_words="english")
vectors = cv.fit_transform(movies["Tags"]).toarray()
similarity_matrix = cosine_similarity(vectors)

def recommend(movie):
    movie_index = movies[movies ["names"] == movie].index[0] # Index of Movie
    distances = similarity_matrix[movie_index] # Distances of movie from other movies
    SimilarMovies = sorted(list(enumerate(distances)), reverse=True, key = lambda x : x[1])[1:6] # Key tells that we want to sort on the basis of 2nd number of the tuple in the list of tuples.

    for i in SimilarMovies:
        print(movies.iloc[i[0]].names)

recommend("Attack on Titan")