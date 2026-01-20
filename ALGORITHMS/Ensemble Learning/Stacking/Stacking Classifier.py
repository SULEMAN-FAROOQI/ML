from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier , StackingClassifier
from sklearn.neighbors import KNeighborsClassifier
import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

x, y = load_digits(return_X_y=True, as_frame=True)

trainx, testx ,trainy ,testy = train_test_split(x,y ,random_state=33, test_size=0.3)

estimators = [
    ("logrg",LogisticRegression(solver="saga", max_iter=5000)),
    ("dtc",DecisionTreeClassifier()),
    ("knnc",KNeighborsClassifier())
]

# Only LogisticRegression supports decision_function(). So decision_function() is not used.

stack1 = StackingClassifier(
    estimators=estimators, 
    final_estimator=RandomForestClassifier(n_estimators=130),
    passthrough=False,
    stack_method="auto"
    )

stack2 = StackingClassifier(
    estimators=estimators, 
    final_estimator=RandomForestClassifier(n_estimators=130),
    passthrough=False,
    stack_method="predict"
    )

stack3 = StackingClassifier(
    estimators=estimators, 
    final_estimator=RandomForestClassifier(n_estimators=130),
    passthrough=False,
    stack_method="predict_proba"
    )

stack4 = StackingClassifier(
    estimators=estimators, 
    final_estimator=RandomForestClassifier(n_estimators=130),
    passthrough=True
    )

# Passthrough = meta is trained on predictions made by base models and training data both.

stack1.fit(trainx,trainy)
stack1_predy = stack1.predict(testx)
print("The Accuracy Score after using auto will be",accuracy_score(testy,stack1_predy))

stack2.fit(trainx,trainy)
stack2_predy = stack2.predict(testx)
print("The Accuracy Score after using predict will be",accuracy_score(testy,stack2_predy))

stack3.fit(trainx,trainy)
stack3_predy = stack3.predict(testx)
print("The Accuracy Score after using predict proba will be",accuracy_score(testy,stack3_predy))

stack4.fit(trainx,trainy)
stack4_predy = stack4.predict(testx)
print("The Accuracy Score after using pass through value true is:",accuracy_score(testy,stack4_predy))
