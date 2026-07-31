import os
import warnings

warnings.filterwarnings('ignore')
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"     # Suppress INFO, WARNING, and ERROR logs from TF/absl
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import tensorflow as tf # type: ignore
from tensorflow.keras import layers # type:ignore
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, AveragePooling2D, Concatenate, Dropout, BatchNormalization, GlobalAveragePooling2D # type:ignore
from tensorflow.keras.models import Model # type:ignore
from scikeras.wrappers import KerasClassifier

tf.get_logger().setLevel('ERROR')            # Belt-and-suspenders: also mute TF's own logger

x = np.zeros((1, 180, 180, 3))
y = np.zeros((1,))

def build_model(input_shape=(180, 180, 3), classes=1, include_aux=False):

    def inception_module(x, filters, name=None):
        a = Conv2D(filters['1x1'], kernel_size=(1,1), strides=1, activation="relu", name=name + '_1x1')(x)

        b1 = Conv2D(filters['3x3_reduce'], kernel_size=(1,1), strides=1, activation="relu", name=name + '_3x3_reduce')(x)
        b2 = Conv2D(filters['3x3'], kernel_size=(3,3), padding="same", strides=1, activation="relu", name=name + '_3x3')(b1)

        c1 = Conv2D(filters['5x5_reduce'], kernel_size=(1,1), strides=1, activation="relu", name=name + '_5x5_reduce')(x)
        c2 = Conv2D(filters['5x5'], kernel_size=(5,5), strides=1, padding="same", activation="relu", name=name + '_5x5')(c1)

        d1 = MaxPool2D(pool_size=(3,3), strides=1, padding="same", name=name + '_pool')(x)
        d2 = Conv2D(filters['pool'], kernel_size=(1,1), strides=1, activation="relu", name=name + '_pool_proj')(d1) 

        output = Concatenate(axis=-1, name=name + '_concat')([a, b2, c2, d2])
        return output

    def auxiliary_classifier(x, classes, name=None):
        aux = AveragePooling2D((5, 5), strides=(3, 3), name=name + '_avgpool')(x)
        aux = Conv2D(128, (1, 1), padding='same', activation='relu', name=name + '_conv')(aux)
        aux = Flatten(name=name + '_flatten')(aux)
        aux = Dense(1024, activation='relu', name=name + '_ANN1')(aux)
        aux = Dropout(0.7, name=name + '_dropout')(aux)
        aux = Dense(classes, activation='sigmoid', name=name + '_output')(aux)
        return aux

    def build_googlenet(input_shape, classes, include_aux):
        inputs = layers.Input(shape=input_shape, name='input')

        x = Conv2D(64, (7, 7), strides=(2, 2), padding='same', activation='relu', name='conv1_7x7_s2')(inputs)
        x = MaxPool2D((3, 3), strides=(2, 2), padding='same', name='pool1_3x3_s2')(x)
        x = BatchNormalization(name='lrn1')(x)

        x = Conv2D(64, (1, 1), padding='same', activation='relu', name='conv2_3x3_reduce')(x)
        x = Conv2D(192, (3, 3), padding='same', activation='relu', name='conv2_3x3')(x)
        x = BatchNormalization(name='lrn2')(x)
        x = MaxPool2D((3, 3), strides=(2, 2), padding='same', name='pool2_3x3_s2')(x)

        # Inception block 3:

        x = inception_module(x, {'1x1': 64, '3x3_reduce': 96, '3x3': 128, '5x5_reduce': 16, '5x5': 32, 'pool': 32}, name='inception3a')
        x = inception_module(x, {'1x1': 128, '3x3_reduce': 128, '3x3': 192, '5x5_reduce': 32, '5x5': 96, 'pool': 64}, name='inception3b')
        x = MaxPool2D((3, 3), strides=(2, 2), padding='same', name='pool3_3x3_s2')(x)

        # Inception block 4:

        x = inception_module(x, {'1x1': 192, '3x3_reduce': 96, '3x3': 208, '5x5_reduce': 16, '5x5': 48, 'pool': 64}, name='inception4a')

        aux1 = auxiliary_classifier(x, classes, name='aux1') if include_aux == True else None

        x = inception_module(x, {'1x1': 160, '3x3_reduce': 112, '3x3': 224, '5x5_reduce': 24, '5x5': 64, 'pool': 64}, name='inception4b')
        x = inception_module(x, {'1x1': 128, '3x3_reduce': 128, '3x3': 256, '5x5_reduce': 24, '5x5': 64, 'pool': 64}, name='inception4c')
        x = inception_module(x, {'1x1': 112, '3x3_reduce': 144, '3x3': 288, '5x5_reduce': 32, '5x5': 64, 'pool': 64}, name='inception4d')

        aux2 = auxiliary_classifier(x, classes, name='aux2') if include_aux == True else None

        x = inception_module(x, {'1x1': 256, '3x3_reduce': 160, '3x3': 320, '5x5_reduce': 32, '5x5': 128, 'pool': 128}, name='inception4e')
        x = MaxPool2D((3, 3), strides=(2, 2), padding='same', name='pool4_3x3_s2')(x)

        # Inception block 5:

        x = inception_module(x, {'1x1': 256, '3x3_reduce': 160, '3x3': 320, '5x5_reduce': 32, '5x5': 128, 'pool': 128}, name='inception5a')
        x = inception_module(x, {'1x1': 384, '3x3_reduce': 192, '3x3': 384, '5x5_reduce': 48, '5x5': 128, 'pool': 128}, name='inception5b')

        x = GlobalAveragePooling2D(name='avgpool_7x7_s1')(x)
        x = Dropout(0.4, name='dropout')(x)
        main_output = Dense(classes, activation='sigmoid', name='main_output')(x)

        if include_aux == True:
            return Model(inputs=inputs, outputs=[main_output, aux1, aux2], name='GoogLeNet_InceptionV1')
        else:
            return Model(inputs=inputs, outputs=main_output, name='GoogLeNet_InceptionV1')

    # build and return, scikeras compiles it using the loss we passed to KerasClassifier
    return build_googlenet(input_shape=input_shape, classes=classes, include_aux=include_aux)

m = KerasClassifier(
    model=build_model,
    model__input_shape=(180, 180, 3),
    model__classes=1,
    model__include_aux=False,
    loss='binary_crossentropy',
    optimizer='adam',
    epochs=10,
    batch_size=32,
)

m.initialize(x, y)
m.model_.summary()