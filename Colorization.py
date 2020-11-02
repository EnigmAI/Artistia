import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from skimage.color import rgb2lab, lab2rgb

import os
import random
import sys
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
import tensorflow.keras.backend as K
from tensorflow.keras.preprocessing import image


def color():
    path = 'static/uploads'
    img_path = os.path.join(path, "source1.png")
    # reconstructed_model = tf.keras.models.load_model("ImageColorization/trained_models_v1/Autoencoder100.hdf5")
    reconstructed_model = tf.keras.models.load_model("ImageColorization/trained_models_v2/U-Net-epoch-100-loss-0.006095.hdf5")
    img = image.img_to_array(image.load_img(img_path))
    h, w = img.shape[0], img.shape[1]
    img_color = []
    img_resize = image.img_to_array(
        image.load_img(img_path, target_size=(256, 256, 3)))
    img_color.append(img_resize)
    img_color = np.array(img_color, dtype=float)
    img_color = rgb2lab(img_color/255.0)[:, :, :, 0]
    img_color = img_color.reshape(img_color.shape+(1,))
    output = reconstructed_model.predict(img_color)
    output = output*128
    result = np.zeros((256, 256, 3))
    result[:, :, 0] = img_color[0][:, :, 0]
    result[:, :, 1:] = output[0]
    final_img = lab2rgb(result)
    fname = "static/results/result_color.png"
    image.save_img(fname, final_img)
