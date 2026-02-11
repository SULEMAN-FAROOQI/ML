from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import catboost as cat
from sklearn.datasets import load_breast_cancer

covid = load_breast_cancer(as_frame=True)

x = covid.data
y = covid.target

trainx ,testx, trainy, testy = train_test_split(x,y, test_size=0.3, random_state=33)

meow1 = cat.CatBoostClassifier(iterations=330, loss_function='Logloss', allow_writing_files=False, silent=True)
meow2 = cat.CatBoostClassifier(iterations = 330, loss_function='CrossEntropy', allow_writing_files=False, silent=True)

# loss_function: 'Logloss' or 'CrossEntropy' for Classification.

meow1.fit(trainx, trainy)
predy_meow1 = meow1.predict(testx)
print("The Accuarcy Score after using Catboost with Logloss is:",accuracy_score(testy,predy_meow1))

meow2.fit(trainx, trainy)
predy_meow2 = meow2.predict(testx)
print("The Accuarcy Score after using Catboost with Cross Entropy is:",accuracy_score(testy,predy_meow2))
