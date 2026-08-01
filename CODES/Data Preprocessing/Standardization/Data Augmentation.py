import os
import warnings

warnings.filterwarnings('ignore')
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"     # Suppress INFO, WARNING, and ERROR logs from TF/absl
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"    # Disable oneDNN custom ops (removes that specific message)

import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image # type: ignore 
# image Behaves like opencv in keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img # type: ignore

datagen = ImageDataGenerator(
        rotation_range=30, # Angle at which we want our image to rotate
        width_shift_range=0.2, # Movement of image verticaly
        height_shift_range=0.2, # Movement of image horizontaly
        shear_range=0.2, # Distortion of image on a scale of 0 to 1 
        zoom_range=0.2, # Random zoom in or zoom out 
        horizontal_flip=True,
        # vertical_flip = True,
        fill_mode='nearest' # 
        )

# fill_mode Strategies Keras:

'''

1. nearest (Default): The empty pixels are filled by extending the color of the closest original edge pixels.
2. reflect: The empty space is filled with a mirrored reflection of the original image across its boundaries.
3. wrap: The empty space is filled by repeating the original image from the opposite edge (like a tiled background).
4. constant: All empty pixels are filled with a single, uniform color value (defined by the cval parameter, which is 0.0 by default, 
resulting in black)

'''

img = image.load_img("Datasets\\Cats vs Dogs\\test\\cats\\cat.10.jpg", target_size=(180,180))
# plt.imshow(img)
# plt.show()

x = img_to_array(img)  # this is a Numpy array with shape (180,180,3)
x = x.reshape(1,180,180,3) 

# the .flow() command below generates batches of randomly transformed images
# and saves the results to the `preview/` directory

i = 0
for batch in datagen.flow(x, batch_size=1, save_to_dir='preview', save_prefix='cat', save_format='jpeg'):
    i += 1
    if i > 20:
        break

# datagen.flow_from_directory is used to perform augmentation on a whole file having images

# Implementation through Scikeras:

'''

import os
import warnings

warnings.filterwarnings('ignore')
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"     # Suppress INFO, WARNING, and ERROR logs from TF/absl
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"    # Disable oneDNN custom ops (removes that specific message)

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import img_to_array, load_img # type: ignore
from tensorflow.keras.layers import RandomFlip, RandomRotation, RandomTranslation, RandomZoom # type: ignore
from tensorflow.keras import Sequential # type: ignore

# augmentation is a small Sequential model made purely of augmentation layers.
# This replaces ImageDataGenerator - each layer below does the same job as one
# of your old ImageDataGenerator parameters, just implemented as a Keras layer.

augmentation = Sequential([
    RandomFlip("horizontal"),          # Same as horizontal_flip=True
    RandomRotation(0.08),              # Angle at which we want our image to rotate (0.08 * 360 ≈ 30 degrees, Keras uses a fraction of a full rotation instead of degrees)
    RandomTranslation(0.2, 0.2),       # Movement of image (height_factor=0.2 = vertical shift, width_factor=0.2 = horizontal shift)
    RandomZoom(0.2),                   # Random zoom in or zoom out, same as zoom_range=0.2
])

# Note: these layers don't have shear_range or fill_mode equivalents built in -
# RandomTranslation/RandomZoom fill empty space using "reflect" by default
# (mirrors the image at the boundary), which can be changed via the
# fill_mode argument on these layers if needed (e.g. fill_mode="constant")

img = load_img("Datasets\\Cats vs Dogs\\test\\cats\\cat.10.jpg", target_size=(180,180))
# plt.imshow(img)
# plt.show()

x = img_to_array(img)   # this is a Numpy array with shape (180,180,3)
x = x.reshape(1,180,180,3)

os.makedirs('preview', exist_ok=True)

# the loop below generates 20 randomly transformed versions of the same image
# and saves the results to the `preview/` directory

# training=True forces the augmentation layers to actually apply randomness -
# by default these layers stay OFF (identity/no-op) unless told they're training,
# since normally they turn themselves off automatically during model.predict()

i = 0
for i in range(20):
    augmented = augmentation(x, training=True)
    augmented_img = augmented[0].numpy().astype("uint8")
    plt.imsave(f"preview/cat_aug_{i}.jpeg", augmented_img)

# augmentation layers like these are meant to be added directly inside your
# build_model() Sequential stack (before the Conv2D layers) so they run
# automatically during scikeras's KerasClassifier.fit() - no flow_from_directory needed

'''

# DATA AUGMENTATION - CLASSES REFERENCE

'''

1. Geometric Transformations
   - Flips (horizontal/vertical), rotation, translation, shear, scale/zoom, crop, perspective/affine warp

2. Photometric / Color Transformations
   - Brightness, contrast, saturation, hue jitter, gamma correction, channel shift, grayscale

3. Noise Injection
   - Gaussian noise, salt-and-pepper, speckle, sensor/ISO noise

4. Blur / Filtering
   - Gaussian blur, motion blur, sharpen, median filter

5. Random Erasing / Occlusion
   - Cutout, Random Erasing, GridMask

6. Mixing-Based
   - Mixup, CutMix, Mosaic

7. Elastic / Distortion-Based
   - Elastic transform, grid distortion, optical distortion

8. Feature-Space Augmentation
   - SMOTE-style interpolation, adversarial perturbation, manifold mixup

9. Generative / Synthetic
   - GAN-generated samples, diffusion-generated samples, style transfer

10. Policy-Based / Automated
    - AutoAugment, RandAugment, TrivialAugment

OTHER MODALITIES (non-image)
- Text: synonym replacement, back-translation, random insert/delete/swap, contextual embedding substitution
- Audio: time-stretch, pitch shift, noise injection, SpecAugment
- Tabular: SMOTE/ADASYN, feature noise, feature permutation
- Time-series: window slicing, jittering, time-warping, magnitude-warping

'''