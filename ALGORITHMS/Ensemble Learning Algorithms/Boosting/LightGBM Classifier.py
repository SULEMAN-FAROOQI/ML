import lightgbm as light
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import fetch_covtype

covid = fetch_covtype(as_frame=True)

x = covid.data
y = covid.target - 1

trainx ,testx, trainy, testy = train_test_split(x,y, test_size=0.3, random_state=33)

lightclass = light.LGBMClassifier(boosting_type="gbdt", max_depth=5, num_leaves=33, n_estimators=300, verbose=-1)
histclass = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.3)

# LightGBM:

'''

Hyperparameters to manage in LightGBM:

1. num_leaves (The most important): This controls the complexity of the model. In a level-wise tree, 
max_depth naturally limits leaves. In LightGBM, if you set a high num_leaves without a max_depth, the model will 
overfit almost instantly.

Num of leaves  = 2 ^ (max_depth)

2. min_data_in_leaf: This is critical for preventing overfitting. It prevents the model from creating a
leaf that only "explains" a tiny handful of samples. On small datasets, setting this to a higher value is 
mandatory.

3. max_depth: Even though LightGBM is leaf-wise, you should still set a depth limit to keep the 
"Ferrari" on the track.

'''

# HistGradientBoosting (Cousion of LightGBM):

'''

The most effective way to tune HistGradientBoosting is the Trade-off Rule. If you decrease the learning_rate by 
half, you should roughly double the max_iter. This allows the model to take smaller, more precise steps 
toward the minimum error.

'''

lightclass.fit(trainx,trainy)
predy_lightclass = lightclass.predict(testx)
print("The Accuracy score after using LightGBM Classifier is:",accuracy_score(testy,predy_lightclass))

histclass.fit(trainx,trainy)
predy_histclass = histclass.predict(testx)
print("The Accuracy score after using Histogram Gradient Boosting Classifier is:",accuracy_score(testy,predy_histclass))
