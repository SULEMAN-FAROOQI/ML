import ast
import numpy as np
import pandas as pd
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity # Similarity is 0 (No Similarity) and 1 (Similar)

pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', 100)

movies = pd.read_csv("C:\\Users\\sulem\\OneDrive\\Desktop\\Codes\\ML\\Datasets\\MovieRecomData\\tmdb_5000_movies.csv")
credits = pd.read_csv("C:\\Users\\sulem\\OneDrive\\Desktop\\Codes\\ML\\Datasets\\MovieRecomData\\tmdb_5000_credits.csv")

# print(movies.head())
# print(credits.head())

movies = movies.merge(credits, on="title")

movies = movies.drop(columns=["budget","id","homepage","original_title","original_language","popularity","production_countries","revenue","release_date","runtime","spoken_languages","status","tagline","vote_average","vote_count"])
movies.dropna(inplace = True)
# print(movies.duplicated().sum())

def NameConverter1(dict):
    L = []
    for i in ast.literal_eval(dict):
        L.append(i["name"])
    return L

def NameConverter2(dict):
    L = []
    counter = 0
    for i in ast.literal_eval(dict):
        if counter != 6:
            L.append(i["name"])
            counter += 1
        else:
            break
    return L

def NameConverter3(dict):
    L = []
    counter = 0
    for i in ast.literal_eval(dict):
        if counter != 3:
            L.append(i["name"])
            counter += 1
        else:
            break
    return L

def NameSearcher(dict):
    L = []
    for i in ast.literal_eval(dict):
        if i["job"] == "Director":
            L.append(i["name"])
            break
    return L

movies["genres"] = movies["genres"].apply(NameConverter1)
movies["keywords"] = movies["keywords"].apply(NameConverter1)
movies["cast"] = movies["cast"].apply(NameConverter2)
movies["production_companies"] = movies["production_companies"].apply(NameConverter3)
movies["Director"] = movies["crew"].apply(NameSearcher)

movies["overview"] = movies["overview"].apply(lambda x : x.split())

movies = movies.drop("crew", axis = 1)

# print(movies.head())

# Removing spaces to convert things into a big string

movies["genres"] = movies["genres"].apply(lambda x : [i.replace(" ","") for i in x])
movies["keywords"] = movies["keywords"].apply(lambda x : [i.replace(" ","") for i in x])
movies["cast"] = movies["cast"].apply(lambda x : [i.replace(" ","") for i in x])
movies["Director"] = movies["Director"].apply(lambda x : [i.replace(" ","") for i in x])
movies["production_companies"] = movies["production_companies"].apply(lambda x : [i.replace(" ","") for i in x])

movies["Tags"] = movies["genres"] + movies["keywords"] + movies["cast"] + movies["Director"] + movies["production_companies"] + movies["overview"]
movies = movies.drop(columns = ["genres","keywords","cast","Director","production_companies","overview"])
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

cv = CountVectorizer(max_features=4000, stop_words="english")
vectors = cv.fit_transform(movies["Tags"]).toarray()

# print(cv.get_feature_names_out())

# For High Dimensions we dont use euclidean distance we use cosine distance.

similarity_matrix = cosine_similarity(vectors)

# This Simalrity matrix contains cosine distance of each movie with all (4806) movies.

def recommend(movie):
    movie_index = movies[movies ["title"] == movie].index[0] # Index of Movie
    distances = similarity_matrix[movie_index] # Distances of movie from other movies
    SimilarMovies = sorted(list(enumerate(distances)), reverse=True, key = lambda x : x[1])[1:6] # Key tells that we want to sort on the basis of 2nd number of the tuple in the list of tuples.

    for i in SimilarMovies:
        print(movies.iloc[i[0]].title)

recommend("Pirates of the Caribbean: At World's End")
