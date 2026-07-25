import os
import warnings

warnings.filterwarnings('ignore')
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"     # Suppress INFO, WARNING, and ERROR logs from TF/absl
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"    # Disable oneDNN custom ops (removes that specific message)

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential # type:ignore
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, BatchNormalization, Dropout, RandomFlip, RandomRotation, RandomZoom, RandomTranslation # type:ignore
from scikeras.wrappers import KerasClassifier
from tensorflow.keras.optimizers import Adam # type:ignore
from tensorflow.keras.preprocessing import image 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay, confusion_matrix

tf.get_logger().setLevel('ERROR')            # Belt-and-suspenders: also mute TF's own logger

train = "C:\\Users\\sulem\\OneDrive\\Desktop\\Codes\\ML\\Datasets\\Cats vs Dogs\\train"

def ImageTransformation(train):

    train_generator = tf.keras.utils.image_dataset_from_directory(
        directory = train,
        labels = "inferred",
        label_mode = "int", # Assigning 0 to cats and 1 to dogs
        batch_size = 33,
        image_size = (180,180)
    )

    # Normalization: Every val in numpy array is between 0 and 255, we will convert it between 0 and 1.

    def Transformation(img, label):
        img = tf.cast(img/255, tf.float32)
        return img, label

    train_pixels = train_generator.map(Transformation)

    return train_pixels

train_pixels = ImageTransformation(train)

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

    input_shape = meta["X_shape_"][1:]
   
# meta["X_shape_"] → (n_samples, 28, 28, 1)

# meta["X_shape_"][1:] → (28, 28, 1) — you slice off the batch dimension, since Keras's input_shape argument wants the shape of a single sample, not the whole batch.

    m = Sequential()

    # Data Augmentation for Overfitting

    m.add(RandomFlip("horizontal", input_shape=input_shape))
    m.add(RandomRotation(0.08))       # ~30 degrees in Keras's [0,1]-of-360 scale
    m.add(RandomTranslation(0.2, 0.2))
    m.add(RandomZoom(0.2))

    m.add(Conv2D(64, kernel_size=(3,3), padding="same", activation="relu"))
    m.add(BatchNormalization())

    m.add(Conv2D(64, kernel_size=(3,3), padding="same", activation="relu"))
    m.add(BatchNormalization())

    m.add(MaxPool2D(pool_size=(2,2), padding = "valid", strides = 2))

    m.add(Conv2D(128, kernel_size=(3,3), padding="same", activation="relu"))
    m.add(BatchNormalization())

    m.add(Conv2D(128, kernel_size=(3,3), padding="same", activation="relu"))
    m.add(BatchNormalization())

    m.add(MaxPool2D(pool_size=(2,2), padding = "valid", strides = 2))

    m.add(Conv2D(256, kernel_size=(3,3), padding="same", activation="relu"))
    m.add(BatchNormalization())

    m.add(Conv2D(256, kernel_size=(3,3), padding="same", activation="relu"))
    m.add(BatchNormalization())

    m.add(Conv2D(256, kernel_size=(3,3), padding="same", activation="relu"))
    m.add(BatchNormalization())

    m.add(MaxPool2D(pool_size=(2,2), padding = "valid", strides = 2))

    m.add(Conv2D(512, kernel_size=(3,3), padding="same", activation="relu"))
    m.add(BatchNormalization())

    m.add(Conv2D(512, kernel_size=(3,3), padding="same", activation="relu"))
    m.add(BatchNormalization())

    m.add(Conv2D(512, kernel_size=(3,3), padding="same", activation="relu"))
    m.add(BatchNormalization())

    m.add(MaxPool2D(pool_size=(2,2), padding = "valid", strides = 2))

    m.add(Conv2D(512, kernel_size=(3,3), padding="same", activation="relu"))
    m.add(BatchNormalization())

    m.add(Conv2D(512, kernel_size=(3,3), padding="same", activation="relu"))
    m.add(BatchNormalization())

    m.add(Conv2D(512, kernel_size=(3,3), padding="same", activation="relu"))
    m.add(BatchNormalization())

    m.add(MaxPool2D(pool_size=(2,2), padding = "valid", strides = 2))

    m.add(Flatten())

    m.add(Dense(4096, activation="relu")) # First layer
    m.add(Dropout(0.5)) # Dropout rate in VGGNET paper is 0.5
    m.add(Dense(4096, activation="relu")) # Second layer
    m.add(Dropout(0.5))
    m.add(Dense(1, activation="sigmoid")) # Output layer

    m.compile(loss="binary_crossentropy", optimizer=Adam(learning_rate = 0.0001), metrics=["accuracy"]) # Using Adam gradient descent as optimizer

    return m

m = KerasClassifier(model=build_model, epochs=10, batch_size=33, verbose=1, validation_split=0.3)

m.fit(trainx,trainy)
predy = m.predict(testx) 
print("Accuarcy Score:", accuracy_score(testy, predy))

# Confusion Matrix
cm = confusion_matrix(testy, predy)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Cat", "Dog"])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix - Test Set")
plt.show()

def predict_image(img_path, model, image_size=(180, 180)):

    img = image.load_img(img_path, target_size=image_size)

    img_array = image.img_to_array(img)
    
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # (1, 180, 180, 3)

    prediction = model.predict(img_array)
    label = "Dog" if prediction[0] == 1 else "Cat"
    print("Prediction:", label)

# Cat Prediction:  
predict_image("C:\\Users\\sulem\\OneDrive\\Desktop\\Codes\\ML\\Datasets\\Cats vs Dogs\\Final_test\\cat0.jpg", m)

# Dog Prediction:  
predict_image("C:\\Users\\sulem\\OneDrive\\Desktop\\Codes\\ML\\Datasets\\Cats vs Dogs\\Final_test\\dog0.jpg", m)

# VGGNET-16 was also originally trained on 14 million images but we are only training on 6k images so the model overfits.