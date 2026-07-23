import os
import warnings

warnings.filterwarnings('ignore')
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"     # Suppress INFO, WARNING, and ERROR logs from TF/absl
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"    # Disable oneDNN custom ops (removes that specific message)

import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import FunctionTransformer
from tensorflow.keras.layers import Dense, Conv2D, AveragePooling2D, Flatten # type: ignore
from tensorflow.keras.models import Sequential # type: ignore
from sklearn.model_selection import train_test_split
from scikeras.wrappers import KerasClassifier 
from sklearn.pipeline import make_pipeline
from tensorflow.keras.callbacks import EarlyStopping  # type: ignore
from sklearn.metrics import accuracy_score

tf.get_logger().setLevel('ERROR')            # Belt-and-suspenders: also mute TF's own logger

data = pd.read_csv("Datasets\\MNIST.csv")
# print(data.head())
# print(data.corr()["label"])

x = data.drop("label", axis = 1)
y = data["label"]

trainx, testx, trainy, testy = train_test_split(x,y, test_size=0.3, random_state=33)

def FeatureTransfrmation(data):

    data = data.values / 255
    data = data.reshape(-1, 28, 28, 1)  # flat pixels -> image format for Conv2D
    return data

f = FunctionTransformer(FeatureTransfrmation)

def build_model(meta):

    input_shape = meta["X_shape_"][1:]
   
# meta["X_shape_"] → (n_samples, 28, 28, 1)

# meta["X_shape_"][1:] → (28, 28, 1) — you slice off the batch dimension, since Keras's input_shape argument wants the shape of a single sample, not the whole batch.

    m = Sequential()

    m.add(Conv2D(6, kernel_size=(5,5), padding="same", activation="tanh", input_shape = input_shape))
    m.add(AveragePooling2D(pool_size=(2,2), padding = "valid", strides = 2))

    m.add(Conv2D(16, kernel_size=(5,5), padding="same", activation="tanh"))
    m.add(AveragePooling2D(pool_size=(2,2), padding = "valid", strides = 2))

    m.add(Flatten())

    m.add(Dense(120, activation="tanh")) # First layer
    m.add(Dense(84, activation="tanh")) # Second layer
    m.add(Dense(10, activation="softmax")) # Output layer

    m.compile(loss="sparse_categorical_crossentropy", optimizer="Adam", metrics=["accuracy"]) # Using Adam gradient descent as optimizer

    return m

callback = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True,
    verbose=0,
)

m = KerasClassifier(model=build_model, epochs=20, batch_size=33, verbose=1, validation_split=0.3, callbacks=[callback])

pipe = make_pipeline(f,m)
pipe.fit(trainx,trainy)

predy = pipe.predict(testx) 
print("Accuracy:", accuracy_score(testy, predy))

sample_x = testx.iloc[[0]]   
sample_y = testy.iloc[0]

pred = pipe.predict(sample_x)
print("Predicted:", pred[0], "| Actual:", sample_y)