import os
import warnings

warnings.filterwarnings('ignore')
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"     # Suppress INFO, WARNING, and ERROR logs from TF/absl

from tensorflow.keras.datasets import imdb # Contains Reviews and Sentiments of Movies
from tensorflow.keras import Sequential, Input # type: ignore
from tensorflow.keras.layers import Dense, SimpleRNN, Flatten, Embedding, Dropout  # type: ignore
from tensorflow.keras.utils import pad_sequences
from scikeras.wrappers import KerasClassifier
from sklearn.metrics import accuracy_score

(trainx, trainy), (testx, testy) = imdb.load_data(num_words=10000)

trainx = pad_sequences(trainx, padding="post", maxlen=180)
testx = pad_sequences(testx, padding="post", maxlen=180)

def build_deep_model(meta):

    m = Sequential() # Here m is our model and now we will add layers in it

    m.add(Input(shape=(180,)))
    m.add(Embedding(input_dim=10000, output_dim=30))

    # Stacked RNN layers : Every layer except the last must return the full sequence (return_sequences=True) so the next RNN layer has a sequence to read timestep by timestep. 
    # Only the final RNN layer collapses to a single vector (return_sequences=False).
                       
    m.add(SimpleRNN(5, return_sequences=True))       
    m.add(SimpleRNN(5, return_sequences=False)) # last layer : only final timestep output

    m.add(Dense(1, activation="sigmoid"))

    m.compile(loss="binary_crossentropy", optimizer="Adam", metrics=["accuracy"])

    return m

m = KerasClassifier(model=build_deep_model, epochs=20, batch_size=33, verbose=1, validation_split=0.3)

m.fit(trainx, trainy)
predy = m.predict(testx)
print("Accuracy:", accuracy_score(testy, predy))

# In this way both DeepLSTM and DeepGRU can be implemented.