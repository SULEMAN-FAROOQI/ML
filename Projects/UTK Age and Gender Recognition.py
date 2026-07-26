import os
import warnings

warnings.filterwarnings('ignore')
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"     # Suppress INFO, WARNING, and ERROR logs from TF/absl

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.base import BaseEstimator
from tensorflow.keras.applications.vgg16 import VGG16 # type:ignore
from tensorflow.keras.layers import Dense, Flatten, Dropout, Input, RandomFlip, RandomRotation, RandomZoom, RandomContrast # type:ignore
from tensorflow.keras.regularizers import l2 # type: ignore
from tensorflow.keras.models import Model # type:ignore
from tensorflow.keras.optimizers import Adam # type:ignore
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.preprocessing import image 
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from tensorflow.keras.callbacks import EarlyStopping # type:ignore

tf.get_logger().setLevel('ERROR')            # Belt-and-suspenders: also mute TF's own logger

filepath = "Datasets\\UTK Face\\train"
scaler = StandardScaler()

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

Age_trainy_scaled = scaler.fit_transform(Age_trainy.reshape(-1,1)).ravel()  
Age_testy_scaled = scaler.transform(Age_testy.reshape(-1,1)).ravel()         

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

def build_model():

    inputs = Input(shape=(180,180,3))

    # Data augmentation -- Active only during training; Keras skips them automatically at prediction.

    x = RandomFlip("horizontal")(inputs)        # faces are left-right symmetric
    x = RandomRotation(0.05)(x)                 # ~18 degrees, small enough to keep faces recognizable
    x = RandomZoom(0.1)(x)
    x = RandomContrast(0.1)(x)                  # helps with lighting variation across different photos

    vgg_out = Convolution_layer(x)               # VGG16 now sees augmented images, not the raw ones
    flatten = Flatten()(vgg_out)

    shared = Dense(512, activation="relu", kernel_regularizer=l2(0.0001))(flatten)   
    shared = Dropout(0.2)(shared)                                                    
    shared = Dense(512, activation="relu", kernel_regularizer=l2(0.0001))(shared)
    shared = Dropout(0.2)(shared)

    age_branch = Dense(256, activation="relu")(shared)  
    age_branch = Dropout(0.3)(age_branch)
    age_branch = Dense(128, activation="relu")(age_branch)   
    age_branch = Dropout(0.2)(age_branch)

    gender_branch = Dense(128, activation="relu")(shared)
    gender_branch = Dropout(0.2)(gender_branch)

    output1 = Dense(1, activation="linear", name="Age")(age_branch)
    output2 = Dense(1, activation="sigmoid", name="Gender")(gender_branch)

    model = Model(inputs=inputs, outputs=[output1, output2])

    model.compile(
        optimizer = Adam(learning_rate = 0.00001),
        loss = {'Age': 'mse', 'Gender': 'binary_crossentropy'},
        metrics = {'Age': 'mae', 'Gender': 'accuracy'},
        loss_weights={'Age': 3, 'Gender': 1}
    )

    return model


class KerasEstimator(BaseEstimator):

    def __init__(self, epochs, batch_size=64):
        self.epochs = epochs
        self.batch_size = batch_size

    def fit(self, X, y):
        X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.15, random_state=33)
        callback = EarlyStopping(monitor="val_Age_mae", mode="min", patience=3, restore_best_weights=True)

        self.model_ = build_model()
        self.model_.fit(
            X_tr, [y_tr[:, 0], y_tr[:, 1]],
            validation_data=(X_val, [y_val[:, 0], y_val[:, 1]]),
            epochs=self.epochs, batch_size=self.batch_size,
            callbacks=[callback], verbose=1)
        
        return self

    def predict(self, X):
        age_pred, gender_pred = self.model_.predict(X)
        gender_labels = (gender_pred > 0.5).astype(int)
        return np.column_stack([age_pred.ravel(), gender_labels.ravel()])

m = KerasEstimator(epochs=20, batch_size=64)

trainy = np.column_stack([Age_trainy_scaled, Gender_trainy])  
m.fit(trainx, trainy)

preds = m.predict(testx)
pred_Age, pred_Gender_labels = preds[:, 0], preds[:, 1]

print("Age R2 Score:", r2_score(Age_testy_scaled, pred_Age))   # scaled, not raw
print("Gender Accuracy Score:", accuracy_score(Gender_testy, pred_Gender_labels))

def predict_image(img_path, model, image_size=(180, 180)):

    img = image.load_img(img_path, target_size=image_size)

    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # (1, 180, 180, 3)

    pred_Age, pred_Gender = model.predict(img_array)   # two separate outputs, in outputs=[output1, output2] order

    Age_prediction = scaler.inverse_transform([[pred_Age[0][0]]])[0][0]
    Gender_prediction = "Female" if pred_Gender[0][0] > 0.5 else "Male"   # UTK Face convention: 0=male, 1=female

    print("Predicted Age:", round(Age_prediction))
    print("Predicted Gender:", Gender_prediction)

# Example:
predict_image("C:\\Users\\sulem\\OneDrive\\Desktop\\Codes\\ML\\Datasets\\UTK Face\\Final_test\\middle_age_man_54.jpg", m.model_)