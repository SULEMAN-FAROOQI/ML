from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import catboost as cat
from sklearn.datasets import fetch_california_housing

house = fetch_california_housing(as_frame=True)

x = house.data
y = house.target

trainx ,testx, trainy, testy = train_test_split(x,y, test_size=0.3, random_state=33)

meow1 = cat.CatBoostRegressor(iterations=330, loss_function='RMSE', allow_writing_files=False, silent=True)
meow2 = cat.CatBoostRegressor(iterations = 330, loss_function='MAE', allow_writing_files=False, silent=True)

# loss_function: 'RMSE' or 'MAE' for Regression.

meow1.fit(trainx, trainy)
predy_meow1 = meow1.predict(testx)
print("The R2 Score after using Catboost with RMSE is:",r2_score(testy,predy_meow1))

meow2.fit(trainx, trainy)
predy_meow2 = meow2.predict(testx)
print("The R2 Score after using Catboost with MAE is:",r2_score(testy,predy_meow2))

# To see which features influenced the house prices most:
print(meow1.get_feature_importance(prettified=True))

'''

MAE model is more robust to outliers. If your house price data had extreme "outlier" mansions, 
the MAE model would likely be more stable, even if its R2 is slightly lower.

'''