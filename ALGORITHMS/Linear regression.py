# By Ordinary Least Squares Method:

# Simple linear regression:

'''

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

d = load_diabetes()

x = d.data[:, 2].reshape(-1, 1)
x = scaler.fit_transform(x)
y = d.target

trainx, testx, trainy, testy = train_test_split(x,y,test_size=0.3,random_state=30)

linear = LinearRegression()
linear.fit(trainx,trainy)
predy = linear.predict(testx)

plt.scatter(x, y)
plt.plot(trainx,linear.predict(trainx),color="Red")
plt.show()

print("The Mean absolute error is:", (mean_absolute_error(testy,predy)))
print("The Mean squared error is:" ,(mean_squared_error(testy,predy)))
print("The Root mean square error is" ,(np.sqrt(mean_squared_error(testy,predy))))
print("The R2 score is:", (r2_score(testy,predy)))

r2 = r2_score(testy,predy)

n = testx.shape[0]
p = 1  # one feature

a = 1 - (1 - r2) * (n - 1) / (n - p - 1)

print("The Adjusted r2 score is:" ,(a)) 

'''

# Multiple Linear Regression:

'''

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

x,y = load_diabetes(return_X_y=True)

trainx, testx, trainy, testy = train_test_split(x,y,test_size=0.3,random_state=30)

linear = LinearRegression()
linear.fit(trainx,trainy)
predy = linear.predict(testx)

print("The Mean absolute error is:", (mean_absolute_error(testy,predy)))
print("The Mean squared error is:" ,(mean_squared_error(testy,predy)))
print("The Root mean square error is" ,(np.sqrt(mean_squared_error(testy,predy))))
print("The R2 score is:", (r2_score(testy,predy)))

r2 = r2_score(testy,predy)

n = testx.shape[0]
p = 10  # 10 features in diabetes dataset

a = 1 - (1 - r2) * (n - 1) / (n - p - 1)

print("The Adjusted r2 score is:" ,(a)) 

'''

# By Stochastic Gradient Descent:
# When using gradient descent the features should be scaled for best results.
# eta0 is the value of learning rate and learning_rate is the condition for it like whether it should change or not.
# It also has penalty attribute which corresponds to regularization technique like L2 , L1 and elasticnet.

'''

import random
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDRegressor
from sklearn.datasets import load_diabetes
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

x,y = load_diabetes(return_X_y=True)

scaler = StandardScaler()
x = scaler.fit_transform(x)

trainx,testx,trainy,testy = train_test_split(x,y,test_size=0.3,random_state=30)

class BGDRegressor:
    
    def __init__(self,learning_rate=0.01,epochs=1000):
        
        self.coef_ = None
        self.intercept_ = None
        self.lr = learning_rate
        self.epochs = epochs
        
    def fit(self,X_train,y_train):
    
        self.intercept_ = 0
        self.coef_ = np.ones(X_train.shape[1])
        
        for i in range(self.epochs):
        
            y_hat = np.dot(X_train,self.coef_) + self.intercept_

            intercept_der = -2 * np.mean(y_train - y_hat)
            self.intercept_ = self.intercept_ - (self.lr * intercept_der)
            
            coef_der = -2 * np.dot((y_train - y_hat),X_train)/X_train.shape[0]
            self.coef_ = self.coef_ - (self.lr * coef_der)
    
    def predict(self,X_test):
        return np.dot(X_test,self.coef_) + self.intercept_

class MBGDRegressor:
    
    def __init__(self,batch_size,learning_rate=0.01,epochs=1000):
        
        self.coef_ = None
        self.intercept_ = None
        self.lr = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        
    def fit(self,X_train,y_train):

        self.intercept_ = 0
        self.coef_ = np.ones(X_train.shape[1])
        
        for i in range(self.epochs):
            
            for j in range(int(X_train.shape[0]/self.batch_size)):
                
                idx = random.sample(range(X_train.shape[0]),self.batch_size)
                
                y_hat = np.dot(X_train[idx],self.coef_) + self.intercept_

                intercept_der = -2 * np.mean(y_train[idx] - y_hat)
                self.intercept_ = self.intercept_ - (self.lr * intercept_der)

                coef_der = (-2 * np.dot((y_train[idx] - y_hat), X_train[idx])) / self.batch_size
                self.coef_ = self.coef_ - (self.lr * coef_der)
    
    def predict(self,X_test):
        return np.dot(X_test,self.coef_) + self.intercept_
    
sgd = SGDRegressor(max_iter=10000,learning_rate="constant",random_state=None,eta0=0.01)
bgd = BGDRegressor()
mbgd = MBGDRegressor(batch_size=33)

bgd.fit(trainx,trainy)
bgd_predy = bgd.predict(testx)
print("The R2 score after Batch Gradient Descent is:",(r2_score(testy,bgd_predy)))

sgd.fit(trainx,trainy)
sgd_predy = sgd.predict(testx)
print("The R2 score after Stochastic Gradient Descent is:",(r2_score(testy,sgd_predy)))

mbgd.fit(trainx,trainy)
mbgd_predy = mbgd.predict(testx)
print("The R2 score after Mini Batch Gradient Descent is:",(r2_score(testy,mbgd_predy)))

'''