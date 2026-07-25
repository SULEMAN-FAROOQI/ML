import os
import warnings

warnings.filterwarnings('ignore')
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"     # Suppress INFO, WARNING, and ERROR logs from TF/absl

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.base import BaseEstimator
from tensorflow.keras.applications.vgg16 import VGG16 # type:ignore
from tensorflow.keras.layers import Dense, Flatten, Dropout # type:ignore
from tensorflow.keras.regularizers import l2 # type: ignore
from tensorflow.keras.models import Model # type:ignore
from tensorflow.keras.optimizers import Adam # type:ignore
from tensorflow.keras.preprocessing import image 
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score

tf.get_logger().setLevel('ERROR')            # Belt-and-suspenders: also mute TF's own logger

filepath = "Datasets\\UTK Face\\train"

age = []
gender = []
img = []

for file in os.listdir(filepath):
    age.append(int(file.split("_")[0]))
    gender.append(int(file.split("_")[1]))
    img.append(file)

df = pd.DataFrame({"Age":age, "Gender":gender, "imgpath":img})

datagen = image.ImageDataGenerator(rescale=1./255)

generator = datagen.flow_from_dataframe(
    dataframe = df,
    directory = filepath,
    x_col = 'imgpath',
    y_col = ['Age', 'Gender'],
    class_mode = 'multi_output',
    batch_size = len(df),   # whole subset in one batch, no accumulation loop needed
    target_size = (180,180),
    shuffle = False
)

x, (Age_y, Gender_y) = generator[0]   # multi_output labels come back as a tuple
print(x.shape, Age_y.shape, Gender_y.shape)

trainx, testx, Age_trainy, Age_testy, Gender_trainy, Gender_testy = train_test_split(x, Age_y, Gender_y, test_size=0.3, random_state=33)

Convolution_layer = VGG16(
    weights = "imagenet",
    include_top = False,
    input_shape = (180,180,3)
)

Convolution_layer.trainable = False

def build_model():

    output = Convolution_layer.layers[-1].output
    flatten = Flatten()(output)

    dense1 = Dense(512, activation="relu", kernel_regularizer=l2(0.001))(flatten)
    drop1  = Dropout(0.3)(dense1)
    dense2 = Dense(512, activation="relu", kernel_regularizer=l2(0.001))(flatten)
    drop2  = Dropout(0.3)(dense2)

    dense3 = Dense(512, activation="relu", kernel_regularizer=l2(0.001))(drop1)
    drop3  = Dropout(0.3)(dense3)
    dense4 = Dense(512, activation="relu", kernel_regularizer=l2(0.001))(drop2)
    drop4  = Dropout(0.3)(dense4)

    dense5 = Dense(256, activation="relu", kernel_regularizer=l2(0.001))(drop3)
    drop5  = Dropout(0.2)(dense5)
    dense6 = Dense(256, activation="relu", kernel_regularizer=l2(0.001))(drop4)
    drop6  = Dropout(0.2)(dense6)

    output1 = Dense(1, activation="linear", name="Age")(drop5)
    output2 = Dense(1, activation="sigmoid", name="Gender")(drop6)

    model = Model(inputs=Convolution_layer.input, outputs=[output1, output2])

    model.compile(
        optimizer = Adam(learning_rate = 0.0001),
        loss = {'Age': 'mse', 'Gender': 'binary_crossentropy'},
        metrics = {'Age': 'mae', 'Gender': 'accuracy'},
        loss_weights={'Age': 3, 'Gender': 99}
    )

    return model

class KerasEstimator(BaseEstimator):

    def __init__(self, epochs=10, batch_size=64):
        self.epochs = epochs
        self.batch_size = batch_size

    def fit(self, X, y):
        # y expected as a 2-column array: column 0 = Age, column 1 = Gender
        self.model_ = build_model()
        self.model_.fit(
            X, [y[:, 0], y[:, 1]],
            epochs=self.epochs, batch_size=self.batch_size, verbose=1
        )
        return self

    def predict(self, X):
        age_pred, gender_pred = self.model_.predict(X)
        gender_labels = (gender_pred > 0.5).astype(int)
        return np.column_stack([age_pred.ravel(), gender_labels.ravel()])

m = KerasEstimator(epochs=10, batch_size=64)

trainy = np.column_stack([Age_trainy, Gender_trainy])

m.fit(trainx, trainy)

preds = m.predict(testx)
pred_Age, pred_Gender_labels = preds[:, 0], preds[:, 1]

print("Age R2 Score:", r2_score(Age_testy, pred_Age))
print("Gender Accuracy Score:", accuracy_score(Gender_testy, pred_Gender_labels))

def predict_image(img_path, model, image_size=(180, 180)):

    img = image.load_img(img_path, target_size=image_size)

    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # (1, 180, 180, 3)

    pred_Age, pred_Gender = model.predict(img_array)   # two separate outputs, in outputs=[output1, output2] order

    Age_prediction = pred_Age[0][0]
    Gender_prediction = "Male" if pred_Gender[0][0] > 0.5 else "Female"   # UTK Face convention: 0=male, 1=female

    print("Predicted Age:", round(Age_prediction))
    print("Predicted Gender:", Gender_prediction)

# Example:
predict_image("C:\\Users\\sulem\\OneDrive\\Desktop\\Codes\\ML\\Datasets\\UTK Face\\Final_test\\middle_age_man_54.jpg", m.model_)