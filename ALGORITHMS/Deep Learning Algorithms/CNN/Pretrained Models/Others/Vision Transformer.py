import os
import warnings

warnings.filterwarnings('ignore')
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"     # Suppress INFO, WARNING, and ERROR logs from TF/absl

import keras
import keras_hub

classifier = keras_hub.models.ImageClassifier.from_preset(
    "vit_base_patch16_224_imagenet"
)

img = keras.utils.load_img("Datasets\\Cats vs Dogs\\Final_test\\dog0.jpg", target_size=(224, 224))
x = keras.utils.img_to_array(img)

preds = classifier.predict([x])
print("\n")
print(preds)