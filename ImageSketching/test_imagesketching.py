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

img_path = 'ImageSketching/face.png'


def dice_coef(y_true, y_pred, smooth=1):
    """
    Dice = (2*|X & Y|)/ (|X|+ |Y|)
         =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
    ref: https://arxiv.org/pdf/1606.04797v1.pdf
    """
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    return (2. * intersection + smooth) / (K.sum(K.square(y_true), -1) + K.sum(K.square(y_pred), -1) + smooth)


def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)


reconstructed_model = tf.keras.models.load_model("ImageSketching/trained_models/model20.hdf5", 
                                                custom_objects = {'dice_coef_loss': dice_coef_loss})

img = image.load_img(img_path, target_size=(256, 256, 3))
plt.imshow(img)
plt.show()

img = image.img_to_array(img)
img = img/255.
img = np.expand_dims(img, axis=0)

output = reconstructed_model.predict(img)
img = output[0]
img = cv2.resize(img, (256, 256))
img = img*255
plt.imshow(img)
plt.show()
