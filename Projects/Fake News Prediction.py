# This code represnets an example of a ML workflow. It classifies whether a news is fake or real.

import nltk
import string
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer 
from sklearn.preprocessing import FunctionTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.metrics import accuracy_score , precision_score 

ps = PorterStemmer()
Tf = TfidfVectorizer()

def ensure_nltk_data():
    resources = {
        "tokenizers/punkt_tab": "punkt_tab",
        "corpora/stopwords": "stopwords",
    }
    for path, name in resources.items():
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(name)

ensure_nltk_data()

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_colwidth', 100)

data = pd.read_csv("Datasets\\News.csv", encoding="latin-1")

# print(data.sample(20))

x = data[["Headline"]]
y = data["Target"]

def TextTransformation(text):

    text = text.lower()
    text = nltk.word_tokenize(text)

    Text = []
    for i in text:
        if i.isalnum():
            Text.append(i)

    TEXT = []
    for j in Text:
        if j not in stopwords.words("english") and j not in string.punctuation:
            TEXT.append(j)

    Sentences = []
    for k in TEXT:
        Sentences.append(ps.stem(k))

    return " ".join(Sentences)

def text_series(text):
    return text.apply(TextTransformation)

T = FunctionTransformer(text_series)

z = make_column_transformer(
    (make_pipeline(T, Tf), "Headline"),
    remainder="drop"
)

trainx,testx,trainy,testy = train_test_split(x,y, test_size=0.3, random_state=33)

m = LogisticRegression(class_weight="balanced") 

# By using Logistic Regression every Fake example in the loss function gets counted as if it were ~2.6x more important than a Real example. 
# The optimizer now gets punished much harder for misclassifying Fake headlines, so it stops defaulting to "Real" 
# just because Real is more common.

pipe = make_pipeline(z,m)

pipe.fit(trainx,trainy)
predy = pipe.predict(testx)

print("The Accuracy score after Logistic Regression is:",accuracy_score(testy,predy))
print("The Precision Score after Logistic Regression is:",precision_score(testy,predy))

print("-------------------------------------------------------------------------------------")

new_data = pd.DataFrame({
    "Headline": ["Positive earnings growth hasn't been enough to get PPG Industries (NYSE:PPG) shareholders a favorable return over the last year"]
})

prediction = pipe.predict(new_data)
result = "Fake" if prediction[0] == 0 else "Real"
print("The News is",result)

