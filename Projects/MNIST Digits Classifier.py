import os
import warnings

warnings.filterwarnings('ignore')
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"     # Suppress INFO, WARNING, and ERROR logs from TF/absl

import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import FunctionTransformer
from tensorflow.keras.layers import RandomRotation, RandomTranslation, RandomZoom, Conv2D, MaxPooling2D, Dropout, Dense, Flatten  # type: ignore
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

    m = Sequential()

    m.add(RandomRotation(0.03, input_shape=input_shape))   # ~11 degrees -- kept small so a 6 doesn't drift toward a 9
    m.add(RandomTranslation(0.1, 0.1))
    m.add(RandomZoom(0.1))

    m.add(Conv2D(32, kernel_size=(3,3), activation="relu"))
    m.add(Conv2D(64, kernel_size=(3,3), activation="relu"))
    m.add(MaxPooling2D(pool_size=(2,2)))
    m.add(Dropout(0.3))

    m.add(Flatten())

    m.add(Dense(128, activation="relu"))
    m.add(Dropout(0.5))
    m.add(Dense(10, activation="softmax"))

    m.compile(loss="sparse_categorical_crossentropy", optimizer="Adam", metrics=["accuracy"])

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