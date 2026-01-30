import optuna
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split , cross_val_score
from sklearn.ensemble import RandomForestRegressor , GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score

x , y = load_diabetes(return_X_y=True, as_frame=True)
trainx , testx , trainy , testy = train_test_split(x , y , test_size=0.3, random_state=33)

def system(trial):

    estimators = trial.suggest_int("n_estimators" , 30,300)
    depth = trial.suggest_int("max_depth", 3,30)

    model = RandomForestRegressor(
        n_estimators=estimators,
        max_depth=depth,
        random_state=33
    )
    
    score = cross_val_score(model, trainx ,trainy, cv=3, scoring="r2").mean()
    return(score) # This line is important because otherwise it will return None.

study = optuna.create_study(direction="minimize", sampler = optuna.samplers.TPESampler())
study.optimize(system, n_trials=30)

# Optuna is a dynamic search space which not only allows us to tune hyperparameters but also allows us to check the efficiency of multiple ML Algorithms.
# The above line runs 30 trials to minimize loss in our system.
# The sampler TPE decides which hyper parameter value will be used in the next trial based on the results of past data.
# For classification, the direction parameter will have maximize bcz we maximize the accuracy(classification) and minimize the loss(regression).
# There are multiple samplers available to optimize our efficiency.
# Optuna also allows us to visualize data on each and every step.

print("Best Trials R2 Score:",study.best_trial.value)
print("Best Hyperparameters",study.best_trial.params)

# Best Trials R2 Score: 0.4311738809337145
# Best Hyperparameters: {'n_estimators': 33, 'max_depth': 22}

forest = RandomForestRegressor(n_estimators=33, max_depth=22)
forest.fit(trainx,trainy)
predy = forest.predict(testx)
print("The R2 score after implementing best hyperparameters is:",r2_score(testy,predy))

# For Tuning Algorithms:

'''

def objective(trial):
    # Choose the algorithm to tune
    classifier_name = trial.suggest_categorical('classifier', ['SVM', 'RandomForest', 'GradientBoosting'])

    if classifier_name == 'SVM':
        # SVM hyperparameters
        c = trial.suggest_float('C', 0.1, 100, log=True)
        kernel = trial.suggest_categorical('kernel', ['linear', 'rbf', 'poly', 'sigmoid'])
        gamma = trial.suggest_categorical('gamma', ['scale', 'auto'])

        model = SVR(C=c, kernel=kernel, gamma=gamma, random_state=42)

    elif classifier_name == 'RandomForest':
        # Random Forest hyperparameters
        n_estimators = trial.suggest_int('n_estimators', 50, 300)
        max_depth = trial.suggest_int('max_depth', 3, 20)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)
        bootstrap = trial.suggest_categorical('bootstrap', [True, False])

        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            bootstrap=bootstrap,
            random_state=42
        )

    elif classifier_name == 'GradientBoosting':
        # Gradient Boosting hyperparameters
        n_estimators = trial.suggest_int('n_estimators', 50, 300)
        learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3, log=True)
        max_depth = trial.suggest_int('max_depth', 3, 20)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)

        model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=42
        )

    # Perform cross-validation and return the mean accuracy
    score = cross_val_score(model, trainx, trainy, cv=3, scoring='accuracy').mean()
    return score

'''

# We can also see our trials in a dataframe using study.trials_dataframe().