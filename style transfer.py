from google.colab import drive
drive.mount('/content/drive/')

import numpy as np
from PIL import Image
import requests
from io import BytesIO
from keras import backend
from keras.models import Model
from keras.applications.vgg16 import VGG16
import tensorflow as tf
from scipy.optimize import fmin_l_bfgs_b

ITERATIONS = 12
CHANNELS = 3
IMAGE_SIZE = 400
IMAGE_WIDTH = IMAGE_SIZE
IMAGE_HEIGHT = IMAGE_SIZE
IMAGENET_MEAN_RGB_VALUES = [123.68, 116.779, 103.939]
CONTENT_WEIGHT = 0.04
STYLE_WEIGHT = 5
TOTAL_VARIATION_WEIGHT = 0.9
TOTAL_VARIATION_LOSS_FACTOR = 1.15

input_image_path = "input.png"
style_image_path = "style.png"
output_image_path = "output.png"
combined_image_path = "combined.png"
san_francisco_image_path = "/content/drive/My Drive/input56.png"
tytus_image_path = "/content/drive/My Drive/style56.png"

def load_img(path_to_img):
  max_dim = 512
  img = Image.open(path_to_img)
  long = max(img.size)
  scale = max_dim/long
  img = img.resize((IMAGE_WIDTH,IMAGE_HEIGHT), Image.ANTIALIAS)
  return imgit

input_image = load_img(san_francisco_image_path)
input_image.save(input_image_path)
input_image

style_image = load_img(tytus_image_path)
style_image.save(style_image_path)
style_image

def content_loss(content, combination):
    return backend.sum(backend.square(combination - content))
layers = dict([(layer.name, layer.output) for layer in model.layers])
content_layer = "block2_conv2"
layer_features = layers[content_layer]
content_image_features = layer_features[0, :, :, :]
combination_features = layer_features[2, :, :, :]
loss = backend.variable(0.)
loss = loss + CONTENT_WEIGHT * content_loss(content_image_features,combination_features)

def gram_matrix(x):
    features = backend.batch_flatten(backend.permute_dimensions(x, (2, 0, 1)))
    gram = backend.dot(features, backend.transpose(features))
    return gram

def compute_style_loss(style, combination):
    style = gram_matrix(style)
    combination = gram_matrix(combination)
    size = IMAGE_HEIGHT * IMAGE_WIDTH
    return backend.sum(backend.square(style - combination)) / (4. * (CHANNELS ** 2) * (size ** 2))
style_layers = ["block1_conv2", "block2_conv2", "block3_conv3", "block4_conv3", "block5_conv3"]
for layer_name in style_layers:
    layer_features = layers[layer_name]
    style_features = layer_features[1, :, :, :]
    combination_features = layer_features[2, :, :, :]
    style_loss = compute_style_loss(style_features, combination_features)
    loss += (STYLE_WEIGHT / len(style_layers)) * style_loss

def total_variation_loss(x):
    a = backend.square(x[:, :IMAGE_HEIGHT-1, :IMAGE_WIDTH-1, :] - x[:, 1:, :IMAGE_WIDTH-1, :])
    b = backend.square(x[:, :IMAGE_HEIGHT-1, :IMAGE_WIDTH-1, :] - x[:, :IMAGE_HEIGHT-1, 1:, :])
    return backend.sum(backend.pow(a + b, TOTAL_VARIATION_LOSS_FACTOR))
loss = loss + TOTAL_VARIATION_WEIGHT * total_variation_loss(combination_image)

outputs = [loss]
outputs += backend.gradients(loss, combination_image)
def evaluate_loss_and_gradients(x):
    x = x.reshape((1, IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS))
    outs = backend.function([combination_image], outputs)([x])
    loss = outs[0]
    gradients = outs[1].flatten().astype("float64")
    return loss, gradients
class Evaluator:
    def loss(self, x):
        loss, gradients = evaluate_loss_and_gradients(x)
        self._gradients = gradients
        return loss
    def gradients(self, x):
        return self._gradients
evaluator = Evaluator()

x = np.random.uniform(0, 255, (1, IMAGE_HEIGHT, IMAGE_WIDTH, 3)) - 128.
for i in range(ITERATIONS):
    x, loss, info = fmin_l_bfgs_b(evaluator.loss, x.flatten(), fprime=evaluator.gradients, maxfun=20)
    print("Iteration %d completed with loss %d" % (i, loss)) 

