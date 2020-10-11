import tensorflow as tf
if tf.__version__.startswith('2'):
    tf.compat.v1.disable_eager_execution()

from tensorflow.keras.layers import Input, Lambda, Dense, Flatten
from tensorflow.keras.layers import AveragePooling2D, MaxPooling2D
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing import image
import tensorflow.keras.backend as K

import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import fmin_l_bfgs_b
from datetime import datetime


def VGG16_AvgPool(shape):
    # we want to account for features across the entire image so get rid of the maxpool which throws away information and use average      # pooling instead.
    vgg = VGG16(input_shape=shape, weights='imagenet', include_top=False)

    i = vgg.input
    x = i
    for layer in vgg.layers:
        if layer.__class__ == MaxPooling2D:
            # replace it with average pooling
            x = AveragePooling2D()(x)
        else:
            x = layer(x)

    return Model(i, x)


def VGG16_AvgPool_CutOff(shape, num_convs):
    # this function creates a partial model because we don't need the full VGG network instead we need to stop at an intermediate
    # convolution. Therefore this function allows us to specify how many convolutions we need
    # there are 13 convolutions in total we can pick any of them as the "output" of our content model

    if num_convs < 1 or num_convs > 13:
        print("num_convs must be in the range [1, 13]")
        return None

    model = VGG16_AvgPool(shape)

    n = 0
    output = None
    for layer in model.layers:
        if layer.__class__ == Conv2D:
            n += 1
        if n >= num_convs:
            output = layer.output
            break

    return Model(model.input, output)


# load the content image
def load_img_and_preprocess(path, shape=None):
    img = image.load_img(path, target_size=shape)

    # convert image to array and preprocess for vgg
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    return x


# since VGG accepts BGR this function allows us to convert our values back to RGB so we can plot it using matplotlib
# so this basically reverses the keras function - preprocess input
def unpreprocess(img):
    img[..., 0] += 103.939
    img[..., 1] += 116.779
    img[..., 2] += 126.68
    img = img[..., ::-1]
    return img


def scale_img(x):
    x = x - x.min()
    x = x / x.max()
    return x


def gram_matrix(img):
    # input is (H, W, C) (C = # feature maps)
    # we first need to convert it to (C, H*W)
    X = K.batch_flatten(K.permute_dimensions(img, (2, 0, 1)))

    # now, calculate the gram matrix
    # gram = XX^T / N
    # the constant is not important since we'll be weighting these
    G = K.dot(X, K.transpose(X)) / img.get_shape().num_elements()
    return G


def style_loss(y, t):
    return K.mean(K.square(gram_matrix(y) - gram_matrix(t)))


# function to minimise loss by optimising input image
def minimize(fn, epochs, batch_shape, content_image):
    t0 = datetime.now()
    losses = []
    # x = np.random.randn(np.prod(batch_shape))
    x = content_image
    for i in range(epochs):
        x, l, _ = fmin_l_bfgs_b(
            func=fn,
            x0=x,
            maxfun=20
        )
        x = np.clip(x, -127, 127)
        print("iter=%s, loss=%s" % (i+1, l))
        losses.append(l)

    print("duration:", datetime.now() - t0)
    # plt.plot(losses)
    # plt.show()

    newimg = x.reshape(*batch_shape)
    final_img = unpreprocess(newimg)
    return final_img[0]


if __name__ == "__main__":

    content_path = 'images/content/venice.jpg'
    # style_path = 'images/style/lesdemoisellesdavignon.jpg'
    style_path = 'images/style/starrynight.jpg'

    # we will assume the weight of the content loss is 1 and only weight the style losses
    # style_weights = [0.2, 0.4, 0.3, 0.5, 0.2]
    style_weights = [5, 4, 8, 7, 9]

    x = load_img_and_preprocess(content_path)
    h, w = x.shape[1:3]

    # reduce image size while keeping ratio of dimensions same
    i = 2
    h_new = h
    w_new = w
    while h_new > 400 or w_new > 400:
        h_new = h/i
        w_new = w/i
        i += 1

    h = int(h_new)
    w = int(w_new)

    # print(h, w)

    fig = plt.figure()

    img_content = image.load_img(content_path, target_size=(h, w))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(img_content)

    img_style = image.load_img(style_path, target_size=(h, w))
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.imshow(img_style)

    plt.show()

    # loading and preprocessing input images
    content_img = load_img_and_preprocess(content_path, (h, w))
    style_img = load_img_and_preprocess(style_path, (h, w))

    batch_shape = content_img.shape
    shape = content_img.shape[1:]

    # load the complete model
    vgg = VGG16_AvgPool(shape)

    # load the content model and we only want one output from this which is from 13th layer
    content_model = Model(vgg.input, vgg.layers[13].get_output_at(0))
    # target outputs from content image
    content_target = K.variable(content_model.predict(content_img))

    # index 0 correspond to the original vgg with maxpool so we do get_output_at(1) which corresponds to vgg with avg pool
    # we collect all the convolutional layers in this list because we will need to take output from all of them
    symbolic_conv_outputs = [
        layer.get_output_at(1) for layer in vgg.layers
        if layer.name.endswith('conv1')
    ]

    # make a big model that outputs multiple layers' outputs(outputs from all layers stored in list symbolic_conv_outputs)
    style_model = Model(vgg.input, symbolic_conv_outputs)
    # calculate the targets from convolutional outputs at each layer in symbolic_conv_outputs
    style_layers_outputs = [K.variable(y)
                            for y in style_model.predict(style_img)]

    # create the total loss which is the sum of content + style loss
    loss = K.mean(K.square(content_model.output - content_target))
    for w, symbolic, actual in zip(style_weights, symbolic_conv_outputs, style_layers_outputs):
        # gram_matrix() expects a (H, W, C) as input
        loss += w * style_loss(symbolic[0], actual[0])

    # NOTE: it doesn't matter which model's input you use they are both pointing to the same keras Input layer in memory
    grads = K.gradients(loss, vgg.input)

    get_loss_and_grads = K.function(
        inputs=[vgg.input],
        outputs=[loss] + grads
    )

    def get_loss_and_grads_wrapper(x_vec):
        l, g = get_loss_and_grads([x_vec.reshape(*batch_shape)])
        return l.astype(np.float64), g.flatten().astype(np.float64)

    # converting image shape to 1d array
    img = np.reshape(content_img, (-1))

    final_img = minimize(get_loss_and_grads_wrapper, 10, batch_shape, img)

    plt.imshow(scale_img(final_img))
    plt.show()
