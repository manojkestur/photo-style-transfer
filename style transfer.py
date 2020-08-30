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

ITERATIONS = 15
CHANNELS = 3
IMAGE_SIZE = 400
IMAGE_WIDTH = IMAGE_SIZE
IMAGE_HEIGHT = IMAGE_SIZE
IMAGENET_MEAN_RGB_VALUES = [123.68, 116.779, 103.939]
CONTENT_WEIGHT = 0.06
STYLE_WEIGHT = 5
TOTAL_VARIATION_WEIGHT = 0.89
TOTAL_VARIATION_LOSS_FACTOR = 1.0

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