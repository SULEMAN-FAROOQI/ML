import os
import warnings

warnings.filterwarnings('ignore')
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"     # Suppress INFO, WARNING, and ERROR logs from TF/absl
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"    # Disable oneDNN custom ops (removes that specific message)

import numpy as np
import pandas as pd
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

filepath = 

age = []
gender = []
img = []

for file in os.listdir(filepath):
    age.append(int(file.split("_")[0]))
    gender.append(int(file.split("_")[1]))
    img.append(file)

df = pd.DataFrame({"Age":age, "Gender":gender, "imgpath":img})

