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

# datagen.flow_from_directory is used to perform augmentation on a whole file having imagesS