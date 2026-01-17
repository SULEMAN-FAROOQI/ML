import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression , SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

poly = PolynomialFeatures(degree=2,include_bias=True) 
lr = LinearRegression()
sgd = SGDRegressor(max_iter=10000,learning_rate="constant",random_state=None,eta0=0.01)

x = 5 * np.random.rand(200, 1) - 2.5
y = 0.6 * x**2 + 1.1 * x + 1.5 + 0.8 * np.random.randn(200, 1)
y = y.ravel()

# Our equation for x was of degree 2 so we used degree 2 , if the degree was n then we would have used nth degree.

plt.plot(x,y,"g.") 
plt.title("General Visualization")
plt.show()

# Here g. will gives us green dots instead of lines connected to each green dot.

trainx,testx,trainy,testy = train_test_split(x,y,test_size=0.3,random_state=33)

lr.fit(trainx,trainy)
lr_predy = lr.predict(testx)
print("The R2 score after using linear regression is:",r2_score(testy,lr_predy))

plt.plot(trainx,lr.predict(trainx),"r.")
plt.plot(x,y,"g.")
plt.title("Linear Regression with OLS")
plt.show()

sgd.fit(trainx,trainy)
sgd_predy = sgd.predict(testx)
print("The r2 score after using gradient descent is", r2_score(testy,sgd_predy))

plt.plot(trainx,sgd.predict(trainx),"r.")
plt.plot(x,y,"g.")
plt.title("Linear Regression with Gradient Descent")
plt.show()

poly_x = poly.fit_transform(x)
poly_trainx,poly_testx,trainy,testy = train_test_split(poly_x,y,test_size=0.3,random_state=33)

lr.fit(poly_trainx,trainy)
poly_lr_predy = lr.predict(poly_testx)
print("The R2 score after using Polynomial linear regression is:",r2_score(testy,poly_lr_predy))

plt.plot(poly_trainx,lr.predict(poly_trainx),"r.")
plt.plot(x,y,"g.")
plt.title("Polynomial Linear Regression with OLS")
plt.show()

sgd.fit(poly_trainx,trainy)
sgd_predy = sgd.predict(poly_testx)
print("The r2 score after using Polynomial gradient descent is", r2_score(testy,sgd_predy))

plt.plot(poly_trainx,sgd.predict(poly_trainx),"r.")
plt.plot(x,y,"g.")
plt.title("Polynomial Linear Regression with Gradient Descent")
plt.show()

# The R2 score is consistently changing because the data is generated at random.
# The graph of polynomial regression shows multiple figures because there are multiple columns in trainx and poly_trainx.
