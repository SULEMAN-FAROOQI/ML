import os
import warnings

warnings.filterwarnings('ignore')
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"     # Suppress INFO, WARNING, and ERROR logs from TF/absl

from tensorflow.keras.datasets import imdb # Contains Reviews and Sentiments of Movies
from tensorflow.keras import Sequential, Input # type: ignore
from tensorflow.keras.layers import Dense, SimpleRNN, Flatten, Embedding  # type: ignore
from tensorflow.keras.utils import pad_sequences
from scikeras.wrappers import KerasClassifier
from sklearn.metrics import accuracy_score

(trainx, trainy), (testx, testy) = imdb.load_data(num_words=10000)

trainx = pad_sequences(trainx, padding = "post", maxlen = 180)
testx = pad_sequences(testx, padding = "post", maxlen = 180)

# print(trainx.shape)
# print(testx.shape)

def build_model(meta):

    m = Sequential() # Here m is our model and now we will add layers in it

    m.add(Input(shape=(180,)))               # declares the input shape so Sequential builds immediately
    m.add(Embedding(input_dim=10000, output_dim=30))
    m.add(SimpleRNN(128, return_sequences = False)) # return_sequences = False gives output at the last timestamp, it will be True for a Translator in which we need output on every Timestamp.
    m.add(Dense(1, activation="sigmoid"))

    m.compile(loss="binary_crossentropy", optimizer="Adam", metrics=["accuracy"])  

    return m

m = KerasClassifier(model=build_model, epochs=20, batch_size=33, verbose=1, validation_split=0.3)

m.fit(trainx, trainy)
predy = m.predict(testx) 
print("Accuracy:", accuracy_score(testy, predy))