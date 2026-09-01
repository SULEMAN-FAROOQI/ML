import os
import warnings

warnings.filterwarnings('ignore')
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"     # Suppress INFO, WARNING, and ERROR logs from TF/absl

import numpy as np
import keras
from keras.applications import nasnet

model = nasnet.NASNetLarge(weights="imagenet")

img = keras.utils.load_img("Datasets\\Cats vs Dogs\\Final_test\\dog0.jpg", target_size=(331, 331))
x = keras.utils.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = nasnet.preprocess_input(x)

preds = model.predict(x)
print("\n")
print(nasnet.decode_predictions(preds, top=5)[0]) # Prints Top 5 Pedictions