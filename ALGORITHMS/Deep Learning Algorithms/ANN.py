# Binary Classification ANN:

'''

import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import FunctionTransformer, StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense # type: ignore
from tensorflow.keras.callbacks import EarlyStopping  # type: ignore
from sklearn.metrics import accuracy_score
from scikeras.wrappers import KerasClassifier
import warnings

warnings.filterwarnings('ignore')
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"     # Suppress INFO, WARNING, and ERROR logs from TF/absl
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"    # Disable oneDNN custom ops (removes that specific message)
tf.get_logger().setLevel('ERROR')            # Belt-and-suspenders: also mute TF's own logger

df = pd.read_csv("Datasets\\Churn_Modelling.csv")

# print(df.sample(10))
# print(df.shape)
# print(df.isnull().sum())
# print(df.duplicated().sum())
# print(df.describe())

x = df.drop("Exited", axis = 1)
y = df["Exited"]

trainx, testx, trainy, testy = train_test_split(x, y, test_size=0.3, random_state=33, stratify=y)

def FeaturetTransformation(data):

    data = data.drop(columns = ["RowNumber", "CustomerId", "Surname"])
    return data

f = FunctionTransformer(FeaturetTransformation)

z = make_column_transformer(
    (StandardScaler() , ["CreditScore", "Age", "Balance", "EstimatedSalary"]),
    (OneHotEncoder(sparse_output=False) , ["Geography", "Gender"]),
    remainder = "passthrough"
)

def build_model(meta):

    n_features = meta["n_features_in_"]

    m = Sequential() # Here m is our model and now we will add layers in it

    m.add(Dense(16, activation="relu", input_dim=n_features))  # Hidden Layer 1
    m.add(Dense(8, activation="relu"))  # Hidden Layer 2

    m.add(Dense(1, activation="sigmoid")) # Output Layer

    m.compile(loss="binary_crossentropy", optimizer="Adam", metrics=["accuracy"]) # Using Adam gradient descent as optimizer
    
    return m

callback = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True,
    verbose=0,
)

m = KerasClassifier(model=build_model, epochs=50, batch_size=33, verbose=0, validation_split=0.3, callbacks=callback)

# print(m.summary()) Gives Summary of the models perfomance
# print(m.layers[0].get_weights) Prints weight and biases of 0th layer

pipe = make_pipeline(f, z, m)
pipe.fit(trainx, trainy)
logy = pipe.predict(testx)
predy = np.where(logy > 0.5, 1, 0)

history = pipe.named_steps["kerasclassifier"]
plt.plot(history.history_["loss"], label="train loss")
plt.plot(history.history_["val_loss"], label="val loss")
plt.plot(history.history_["accuracy"], label="accuracy")
plt.plot(history.history_["val_accuracy"], label="val accuracy")
plt.legend()
plt.show()

print("The Accuracy Score is:",accuracy_score(testy,predy))

'''

# MultiNomial Classification ANN:

'''

import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Dropout # type: ignore
from tensorflow.keras.callbacks import EarlyStopping  # type: ignore
from sklearn.metrics import accuracy_score
from scikeras.wrappers import KerasClassifier
import warnings

warnings.filterwarnings('ignore')
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"     # Suppress INFO, WARNING, and ERROR logs from TF/absl
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"    # Disable oneDNN custom ops (removes that specific message)
tf.get_logger().setLevel('ERROR')            # Belt-and-suspenders: also mute TF's own logger

df = pd.read_csv("Datasets\\MNIST.csv")

# print(df.sample(10))
# print(df.shape)

x = df.drop("label", axis = 1)
y = df["label"]

trainx, testx, trainy, testy = train_test_split(x, y, test_size=0.3, random_state=33, stratify = y)

# plt.imshow(trainx.iloc[3].values.reshape(28, 28), cmap="gray")
# plt.show()

def FeatureTransfrmation(data):

    data = data/255
    return data

f = FunctionTransformer(FeatureTransfrmation)

def build_model(meta):

    n_features = meta["n_features_in_"]

    m = Sequential() # Here m is our model and now we will add layers in it

    m.add(Dense(256, activation="relu", input_dim=n_features)) # Input layer
    m.add(Dropout(0.2)) # For Removing Overfitting Factor 
    m.add(Dense(128, activation="relu")) # Hidden layer 1
    m.add(Dropout(0.2))
    m.add(Dense(64, activation="relu")) # Hidden layer 2
    m.add(Dense(10, activation="softmax")) # Output layer

    m.compile(loss="sparse_categorical_crossentropy", optimizer="Adam", metrics=["accuracy"]) # Using Adam gradient descent as optimizer
    
    return m

callback = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True,
    verbose=0,
)

m = KerasClassifier(model=build_model, epochs=50, batch_size=33, verbose=0, validation_split=0.3, callbacks=callback)

pipe = make_pipeline(f, m)
pipe.fit(trainx, trainy)

predy = pipe.predict(testx) 
print("Accuracy:", accuracy_score(testy, predy))

history = pipe.named_steps["kerasclassifier"]
plt.plot(history.history_["loss"], label="train loss")
plt.plot(history.history_["val_loss"], label="val loss")
plt.plot(history.history_["accuracy"], label="accuracy")
plt.plot(history.history_["val_accuracy"], label="val accuracy")
plt.legend()
plt.show()

'''

# ANN Regressor

'''

import os
import numpy as np
import pandas as pd
import tensorflow as tf # type: ignore
import matplotlib.pyplot as plt
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler # Used when we know upper bound and lower bound of values
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Dropout # type: ignore
from sklearn.metrics import r2_score
from tensorflow.keras.regularizers import l2 # type: ignore
from tensorflow.keras.callbacks import EarlyStopping  # type: ignore
from scikeras.wrappers import KerasRegressor
import warnings

warnings.filterwarnings('ignore')
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"     # Suppress INFO, WARNING, and ERROR logs from TF/absl
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"    # Disable oneDNN custom ops (removes that specific message)
tf.get_logger().setLevel('ERROR')            # Belt-and-suspenders: also mute TF's own logger

df = pd.read_csv("Datasets\\Admission_Predict_Ver1.1.csv")
df.columns = df.columns.str.strip()

# print(df.sample(10))
# print(df.shape)
# print(df.duplicated().sum())

x = df.drop("Chance of Admit", axis = 1)
y = df["Chance of Admit"]

trainx, testx, trainy, testy = train_test_split(x, y, test_size=0.3, random_state=33)

def FeatureTransformer(data):

    data = data.drop("Serial No.", axis = 1)
    return data

f = FunctionTransformer(FeatureTransformer)

z = make_column_transformer(
    (MinMaxScaler(), ["GRE Score", "TOEFL Score", "CGPA", "SOP", "LOR"]),
    remainder= "passthrough"
)

def build_model(meta):

    n_features = meta["n_features_in_"]

    m = Sequential() # Here m is our model and now we will add layers in it

    m.add(Dense(16, activation="relu", input_dim=n_features, kernel_regularizer=l2(0.005))) # Input layer 
    m.add(Dropout(0.2)) # For Removing Overfitting Factor 
    m.add(Dense(8, activation="relu", kernel_regularizer=l2(0.005))) # Hidden layer 1
    m.add(Dropout(0.2))
    m.add(Dense(1, activation="linear")) # Output layer

    m.compile(loss="mean_squared_error", optimizer="Adam") # Using Adam gradient descent as optimizer
    
    return m

callback = EarlyStopping(
    monitor="val_loss",
    patience=15,
    restore_best_weights=True,
    verbose=0,
)

m = KerasRegressor(model=build_model, epochs=150, batch_size=33, verbose=0, validation_split=0.3, callbacks=callback)

pipe = make_pipeline(f, z, m)
pipe.fit(trainx, trainy)

predy = pipe.predict(testx) 
print("R2 Score:", r2_score(testy, predy))

history = pipe.named_steps["kerasregressor"]
plt.plot(history.history_["loss"], label="train loss")
plt.plot(history.history_["val_loss"], label="val loss")
plt.legend()
plt.show()

'''

# EarlyStopping Callback — Arguments

'''

monitor
Quantity to be monitored.
Default: "val_loss"

min_delta
Minimum change in the monitored quantity to qualify as an improvement.
An absolute change smaller than min_delta counts as no improvement.
Default: 0

patience
Number of epochs with no improvement after which training will be stopped.
Default: 0

verbose
Verbosity mode: 0 or 1.
0 = silent
1 = displays messages when the callback takes an action
Default: 0

mode
One of {"auto", "min", "max"}
- min: training stops when the monitored quantity stops decreasing
- max: training stops when the monitored quantity stops increasing
- auto: direction is inferred automatically from the name of the monitored quantity
Default: "auto"

baseline
Baseline value for the monitored quantity.
If not None, training stops if the model doesn't improve over this baseline.
Default: None

restore_best_weights
Whether to restore model weights from the epoch with the best value of the monitored quantity.
- If False, uses the weights from the last training step.
- An epoch is restored regardless of performance relative to baseline.
- If no epoch improves on baseline, training runs for `patience` epochs and restores the best epoch from that set.
Default: False

start_from_epoch
Number of epochs to wait before monitoring for improvement begins.
Allows a warm-up period where no improvement is expected, so training isn't stopped prematurely.
Default: 0

'''