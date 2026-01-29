# Log Transform:

'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline

dataframe = pd.read_csv("Datasets\\cars.csv", usecols=["brand" , "km_driven" , "selling_price"])
dataframe.dropna(inplace= True)
# print(dataframe.head())

x = dataframe.iloc[ : , 0:2]
y = dataframe.iloc[: , 2]

# print(dataframe["brand"].value_counts())

# print(x.head())
# print(y.head())

sns.relplot(data=dataframe, x="km_driven", y="selling_price", kind="scatter", hue = "brand")
plt.show()

trainx , testx , trainy, testy = train_test_split(x,y, test_size=0.3, random_state=30)

k = make_column_transformer(
    [OrdinalEncoder(categories=[['Maruti', 'Hyundai', 'Mahindra', 'Tata', 'Toyota', 'Honda', 'Ford',   
       'Chevrolet', 'Renault', 'Volkswagen', 'BMW', 'Skoda', 'Nissan',       
       'Jaguar', 'Volvo', 'Datsun', 'Mercedes-Benz', 'Fiat', 'Audi', 'Lexus',
       'Jeep', 'Mitsubishi', 'Land', 'Force', 'Isuzu', 'Ambassador', 'Kia',  
       'MG', 'Daewoo', 'Ashok', 'Opel', 'Peugeot']]) , ["brand"]] ,
    [StandardScaler(), ["km_driven"]],
    remainder="passthrough"
)

l = DecisionTreeClassifier()

pipeline = make_pipeline(k,l)
# print(pipeline)

pipeline.fit(trainx, trainy)
predy = pipeline.predict(testx)
print("Accuracy score before Function Transformer: " +str(accuracy_score(testy,predy)))

trainx_transformed , testx_transformed, trainy_transformed, testy_transformed = train_test_split(x , y , test_size=0.3 ,random_state=30)

n = make_column_transformer(
    [OrdinalEncoder(categories=[['Maruti', 'Hyundai', 'Mahindra', 'Tata', 'Toyota', 'Honda', 'Ford',   
       'Chevrolet', 'Renault', 'Volkswagen', 'BMW', 'Skoda', 'Nissan',       
       'Jaguar', 'Volvo', 'Datsun', 'Mercedes-Benz', 'Fiat', 'Audi', 'Lexus',
       'Jeep', 'Mitsubishi', 'Land', 'Force', 'Isuzu', 'Ambassador', 'Kia',  
       'MG', 'Daewoo', 'Ashok', 'Opel', 'Peugeot']]) , ["brand"]] ,
    [FunctionTransformer(func = np.log10) , ["km_driven"]],
    remainder="passthrough"
)

pipeline.fit(trainx_transformed, trainy)
predy_transformed = pipeline.predict(testx_transformed)
print("Accuracy score after using function transformer: " +str(accuracy_score(testy, predy_transformed)))

'''

# Box-Cox Transformation and Yeo-Johnson Transformation:

# Box-Cox requires input data to be strictly positive, while Yeo-Johnson supports both positive or negative data.
# By default the zero-mean standardization and unit-variance normalization is applied to the transformed data.

'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PowerTransformer

# This dataset has continious values and accuracy_score function doesnot support continious values

dataframe = pd.read_csv("Datasets\\concrete_data.csv")
# print(dataframe.value_counts())
dataframe.drop_duplicates(inplace=True)
# print(dataframe.head())

x  = dataframe.iloc[ : , 0 : -1]
y = dataframe["Strength"]

trainx , testx , trainy, testy = train_test_split(x,y, test_size=0.3, random_state=30)

reg = LinearRegression()
power = PowerTransformer(method="box-cox")

reg.fit(trainx,trainy)
predy = reg.predict(testx)
print("Accuracy Score before transformation: " +str(r2_score(testy,predy)))

x_transformed = power.fit_transform(x+0.000000001)
trainx_transformed = power.fit_transform(trainx+0.0000001)
testx_transformed = power.transform(testx+0.0000001)
reg.fit(trainx_transformed,trainy)
predy1 = reg.predict(testx_transformed)
print("Accuracy Score After box-cox transformation: " +str(r2_score(testy,predy1)))
print("The cross validation score is: " +str(np.mean(cross_val_score(reg,x_transformed,y,scoring='r2'))))

power1 = PowerTransformer(method="yeo-johnson")
x_transformed1 = power1.fit_transform(x+0.000000001)
trainx_transformed1 = power1.fit_transform(trainx+0.000000001)
testx_transformed1 = power1.fit_transform(testx+0.0000001)
reg.fit(trainx_transformed1,trainy)
predy11 = reg.predict(testx_transformed1)
print("Accuracy Score After yoe-johnson transformation: " +str(r2_score(testy,predy11)))
print("The cross validation score is: " +str(np.mean(cross_val_score(reg,x_transformed,y,scoring='r2'))))

'''

# In the above code, apparently after statandarization the accuracy score dropped