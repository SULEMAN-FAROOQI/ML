# Here Vgg-16 was used for transfer learning

import os
import warnings

warnings.filterwarnings('ignore')
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"     # Suppress INFO, WARNING, and ERROR logs from TF/absl

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential # type:ignore
from tensorflow.keras.layers import Dense, Flatten, Dropout # type:ignore
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input # type:ignore
from tensorflow.keras.optimizers import Adam  # type:ignore
from tensorflow.keras.regularizers import l2 # type:ignore
from scikeras.wrappers import KerasClassifier 
from tensorflow.keras.preprocessing import image
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

Convolution_layer = VGG16(
    weights = "imagenet",
    include_top = False,
    input_shape = (180,180,3)
)

Convolution_layer.trainable = True

set_trainable = False

for layer in Convolution_layer.layers:
    if layer.name == 'block5_conv1':
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False

'''

for layer in Convolution_layer.layers:
    print(layer.name, layer.trainable)

'''
    
data = "C:\\Users\\sulem\\OneDrive\\Desktop\\Codes\\ML\\Datasets\\Helicopter vs Drones\\Train"

def ImageTransformation(train):

    train_generator = tf.keras.utils.image_dataset_from_directory(
        directory = train,
        labels = "inferred",
        label_mode = "int", # Assigning 0 to Helicopter and 1 to Drones
        batch_size = 33,
        image_size = (180,180)
    )

    # Normalization: Every val in numpy array is between 0 and 255, we will convert it between 0 and 1.

    def Transformation(img, label):
        img = preprocess_input(img)   # replaces img/255
        return img, label

    train_pixels = train_generator.map(Transformation)

    return train_pixels

train_pixels = ImageTransformation(data)

def dataset_to_numpy(dataset):

    images = []
    labels = []

    for img_batch, label_batch in dataset:
        images.append(img_batch.numpy())
        labels.append(label_batch.numpy())

    X = np.concatenate(images, axis=0)
    y = np.concatenate(labels, axis=0)

    return X, y

x, y = dataset_to_numpy(train_pixels)

print(x.shape, y.shape)

trainx, testx, trainy, testy = train_test_split(x,y, test_size=0.3, random_state=33, stratify=y)

def build_model(meta):

    m = Sequential()

    m.add(Convolution_layer)
    m.add(Flatten())
    m.add(Dense(128, activation="relu", kernel_regularizer=l2(0.005)))
    m.add(Dropout(0.2))
    m.add(Dense(64, activation="relu", kernel_regularizer=l2(0.005)))
    m.add(Dropout(0.2))
    m.add(Dense(1, activation="sigmoid"))

    m.compile(loss="binary_crossentropy", optimizer=Adam(learning_rate = 0.00001), metrics=["accuracy"]) # Lr should be taken small while fine tuning

    return m

m = KerasClassifier(model=build_model, epochs=10, batch_size=33, verbose=1, validation_split=0.3)

m.fit(trainx,trainy)
predy = m.predict(testx) 
print("Accuarcy Score:", accuracy_score(testy, predy))

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Confusion Matrix
cm = confusion_matrix(testy, predy)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Helicopter", "Drone"])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix - Test Set")
plt.show()

def predict_image(img_path, model, image_size=(180, 180)):

    img = image.load_img(img_path, target_size=image_size)
    img_array = image.img_to_array(img)

    img_array = tf.reshape(img_array, (1,180,180,3)) # Add Batch dimension

    # Same normalization as our Transformation function
    img_array = preprocess_input(img_array)
    
    prediction = model.predict(img_array)
    label = "Helicopter" if prediction[0] == 1 else "Drone"
    print("Prediction:", label)

# Helicopter Prediction:  
predict_image("C:\\Users\\sulem\\OneDrive\\Desktop\\Codes\\ML\\Datasets\\Helicopter vs Drones\\Final test\\heli0.jpg", m)

# Drone Prediction:  
predict_image("C:\\Users\\sulem\\OneDrive\\Desktop\\Codes\\ML\\Datasets\\Helicopter vs Drones\\Final test\\drone0.jpg", m)
