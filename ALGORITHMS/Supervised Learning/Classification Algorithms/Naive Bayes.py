'''

In scikit-learn, the sklearn.naive_bayes module provides different classes based on the mathematical distribution of your features. 
Choosing the right one depends entirely on what your data looks like (e.g., is it heights and weights, word counts, or "Yes/No" answers?).

1. Gaussian Naive Bayes (GaussianNB): 
            It is best for Continuous or Numerical data like temperature, height, or prices. It assumes that the continuous 
values associated with each class are distributed according to a Normal (Gaussian) distribution. Its commonly used in Predicting a  
disease based on blood pressure and age.

2. Multinomial Naive Bayes (MultinomialNB):
            It is best for Discrete counts like Whole numbers (integers). While it technically expects counts, it also works 
well with fractional frequencies like TF-IDF scores. The data follows a multinomial distribution (like rolling a many-sided die).
It is commonly used in Text classification where you count how many times each word appears in a document.

3. Bernoulli Naive Bayes (BernoulliNB):
            It is best for Binary or Boolean data like Data that is either 0 or 1 (Present or Absent). It focuses on whether a
feature is present or not, rather than how many times it occurs. It is commonly used in Spam detection where you only care 
if the word "Free" appears, not how many times it was repeated.

4. Categorical Naive Bayes (CategoricalNB):
            It is best for Categorical features. Categorical features are Discrete categories that don't have a mathematical 
order (e.g., Colors: "Red", "Blue", "Green"). Each feature has its own categorical distribution. Always should encode your 
categories into integers (0, 1, 2...) before passing them to this model.

Here is a breakdown of which module to use for each type of data:

Sno       Module                       Feature Type                                  Example Data
1.        GaussianNB                   Continuous / Decimals                         "98.6°F, 1.75m, $50.25"
2.        MultinomialNB                Counts / Frequencies,"Word counts             (5 'apples' , 2 'oranges')"
3.        BernoulliNB,Binary           (Yes/No)                                      "1 (clicked), 0 (not clicked)"
4.        CategoricalNB                Categories                                    "Sunny" , "Rainy", "Overcast"
5.        ComplementNB                 Imbalanced Counts                             "Similar to Multinomial, but better for skewed datasets"

'''

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

x,y = load_iris(return_X_y=True, as_frame=True)

sns.kdeplot(data=x , x="sepal length (cm)", hue=y)
plt.show()

# Since the data is normally distributed so we will use GaussianNB class.

trainx ,testx ,trainy, testy = train_test_split(x,y, test_size=0.3, random_state=33)

from sklearn.naive_bayes import GaussianNB

naive = GaussianNB()

naive.fit(trainx,trainy)
predy = naive.predict(testx)
print("The Accuracy score is:",accuracy_score(testy,predy))
