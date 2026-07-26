# This code represnets an example of a ML workflow. It classifies whether a message is spam or ham based on the text message recieved.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer 
import string
from sklearn.preprocessing import LabelEncoder , FunctionTransformer
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.metrics import accuracy_score , precision_score # Precision score is important when working with Imbalanced data.

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

encoder = LabelEncoder()
ps = PorterStemmer()
Tf = TfidfVectorizer()

data = pd.read_csv("Datasets\\spam.csv", encoding="latin-1")

def CleanData(data):
    data = data.drop(columns=["Unnamed: 2","Unnamed: 3","Unnamed: 4"])
    data.rename(columns = {"v1" : "Target" , "v2" : "Text"} , inplace=True)
    data["Target"]  = encoder.fit_transform(data["Target"])
    data = data.drop_duplicates()
    return data

data = CleanData(data)

# print(data.head())
# print(data.shape)
# print(data.isnull().sum())
# print(data.duplicated().sum())

# plt.pie(data["Target"].value_counts(), labels=["ham", "spam"], autopct="%0.2f")
# plt.show()

# By Pie chart we can clearly see that our data has more ham values than spam meaning the data is imbalanced.

def FeatureTransformation(data):
    data["Number of Characters"] = data["Text"].apply(len)
    data["Number of Words"] = data["Text"].apply(lambda x : len(nltk.word_tokenize(x)))
    data["Number of Sentences"] = data["Text"].apply(lambda  y: len(nltk.sent_tokenize(y)))
    return data

data = FeatureTransformation(data)

# print(data.head())

# print(data[data["Target"] == 0][["Number of Characters", "Number of Words", "Number of Sentences"]].describe())
# print(data[data["Target"] == 1][["Number of Characters", "Number of Words", "Number of Sentences"]].describe())

# By describe output we can clearly differentiate between the mean of both ham and spam. We can see that spam messages are larger in word count, sentence count and character count.

def TextTransformation(text):
    
    text = text.lower()                                                            # Conversion to Lower case
    text = nltk.word_tokenize(text)                                                # Tokenization: Convering in a series of words seprated by comma

    Text = []

    for i in text:                                                                 # Removing Alpha numeric Characters
        if i.isalnum():
            Text.append(i)

    TEXT = []

    for j in Text:
        if j not in stopwords.words("english") and j not in string.punctuation:    # Removing stopwards(I,is,am,are) and punctuation keywords.
            TEXT.append(j)

    Sentence = []

    for k in TEXT:
        Sentence.append(ps.stem(k))                                                 # Converting Words to their Roots (dancing : danc)

    return " ".join(Sentence)

# TextTransformation("MY PERCENTAGE IS 90%!... I am Dancing right now")

def text_series(text):
    return text.apply(TextTransformation)

T = FunctionTransformer(text_series)

'''

wc_spam = WordCloud(width=600, height=600, min_font_size=10, background_color="White")
wc_ham = WordCloud(width=600, height=600, min_font_size=10, background_color="White")

spam_wordcloud = wc_spam.generate(data[data["Target"] == 1]["Transformed Text"].str.cat(sep = " "))
ham_wordcloud = wc_ham.generate(data[data["Target"] == 0]["Transformed Text"].str.cat(sep = " "))

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(spam_wordcloud)
plt.axis("off")
plt.title("Spam")

plt.subplot(1, 2, 2)
plt.imshow(ham_wordcloud)
plt.axis("off")
plt.title("Ham")

plt.show()


spam_corpus = []
for msg in data[data['Target'] == 1]['Transformed Text'].tolist():
    for word in msg.split():
        spam_corpus.append(word)

ham_corpus = []
for msg in data[data['Target'] == 0]['Transformed Text'].tolist():
    for word in msg.split():
        ham_corpus.append(word)

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
sns.barplot(x=pd.DataFrame(Counter(spam_corpus).most_common(30))[0],
            y=pd.DataFrame(Counter(spam_corpus).most_common(30))[1])
plt.xticks(rotation='vertical')
plt.title("Spam")

plt.subplot(1, 2, 2)
sns.barplot(x=pd.DataFrame(Counter(ham_corpus).most_common(30))[0],
            y=pd.DataFrame(Counter(ham_corpus).most_common(30))[1])
plt.xticks(rotation='vertical')
plt.title("Ham")

plt.show()

'''

# Naive Bayes gives awesome outputs on textual data.

z = make_column_transformer(
    (make_pipeline(T, Tf), "Text"),
    remainder="drop"
)

x = data[["Text"]]   # Dataframe 
y = data["Target"]

trainx,testx,trainy,testy = train_test_split(x,y, test_size=0.3, random_state=33)

m = MultinomialNB()

pipe = make_pipeline(z,m)

pipe.fit(trainx,trainy)
predy = pipe.predict(testx)

print("The Accuracy score after Multinomial Naive Bayes is:",accuracy_score(testy,predy))
print("The Precision Score after Multinomial Naive Bayes is:",precision_score(testy,predy))

print("-------------------------------------------------------------------------------")

new_data = pd.DataFrame({
    "Text": ["WINNER!! As a valued network customer you have been selected to receive a �900 prize reward! "
             "To claim call 09061701461. Claim code KL341. Valid 12 hours only."]
})

prediction = pipe.predict(new_data)
result = "Important" if prediction[0] == 0 else "Spam"
print("The Message recieved is:",result)
