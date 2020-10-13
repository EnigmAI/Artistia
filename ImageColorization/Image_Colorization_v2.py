import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from skimage.color import rgb2lab, lab2rgb

import os
import random
import sys

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
import tensorflow.keras.backend as K
from tensorflow.keras.preprocessing import image

img_path = 'images/1.jpg'

reconstructed_model = tf.keras.models.load_model("trained_models_v2/U-Net-epoch-100-loss-0.006095.hdf5")

img_color = []
img = image.img_to_array(image.load_img(img_path, target_size=(256,256)))
img_color.append(img)
img_color = np.array(img_color, dtype=float)
# print(img_color.shape)
img_color = rgb2lab(img_color/255.0)[:,:,:,0]
# print(img_color.shape)
img_color = img_color.reshape(img_color.shape+(1,))
# print(img_color.shape)

plt.imshow(rgb2lab(img/255.0)[:,:,0])
plt.show()

output = reconstructed_model.predict(img_color)
# print(output.shape)
output = output*128

result = np.zeros((256, 256, 3))
result[:,:,0] = img_color[0][:,:,0]
result[:,:,1:] = output[0]
# print(result.shape)

plt.imshow(lab2rgb(result))
plt.show()