from sklearn.datasets import load_diabetes
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor , StackingRegressor
from sklearn.neighbors import KNeighborsRegressor
import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

x, y = load_diabetes(return_X_y=True, as_frame=True)

trainx, testx ,trainy ,testy = train_test_split(x,y ,random_state=33, test_size=0.3)

estimators = [
    ("logrg",LinearRegression()),
    ("dtc",DecisionTreeRegressor()),
    ("knnc",KNeighborsRegressor())
]

stack1 = StackingRegressor(
    estimators=estimators, 
    final_estimator=RandomForestRegressor(n_estimators=130),
    passthrough=False
    )

stack2 = StackingRegressor(
    estimators=estimators, 
    final_estimator=RandomForestRegressor(n_estimators=130),
    passthrough=True
    )

# Passthrough = meta is trained on predictions made by base models and training data both.

stack1.fit(trainx,trainy)
stack1_predy = stack1.predict(testx)
print("The R2 Score after not passing through will be",r2_score(testy,stack1_predy))

stack2.fit(trainx,trainy)
stack2_predy = stack2.predict(testx)
print("The R2 Score after using passing through value True is",r2_score(testy,stack2_predy))
