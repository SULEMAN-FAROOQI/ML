import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.base import BaseEstimator, RegressorMixin

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_colwidth', 100)

data = pd.read_csv("Datasets\\gld_price_data.csv")
# print(data.head())
# print(data.describe())

x = data.drop(columns=["GLD"])
y = data["GLD"]

trainx , testx , trainy , testy = train_test_split(x, y, test_size=0.3, random_state=33)

def DateRemover(df):
    return df.drop("Date", axis = 1)

f = FunctionTransformer(DateRemover)

z = make_column_transformer(
    (StandardScaler() , ["SPX", "USO", "SLV", "EUR/USD"]),
    remainder="passthrough"
)

class BGDRegressor(BaseEstimator, RegressorMixin):
    
    def __init__(self,learning_rate=0.01,epochs=1000):
        
        self.coef_ = None
        self.intercept_ = None
        self.lr = learning_rate
        self.epochs = epochs
        
    def fit(self,X_train,y_train):

        X_train = np.array(X_train)
        y_train = np.array(y_train)
    
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
    
m = BGDRegressor()

pipe = make_pipeline(f,z,m)

pipe.fit(trainx,trainy)
predy = pipe.predict(testx)
print("The R2 score after Batch Gradient Descent is:",(r2_score(testy,predy)))

new_data = pd.DataFrame({
    "Date":    ["1/2/2008"],
    "SPX":     [1447.160034],
    "USO":     [78.470001],
    "SLV":     [15.18],
    "EUR/USD": [1.471692]
})

prediction = pipe.predict(new_data) 
print("Predicted GLD price:", prediction[0],"$")