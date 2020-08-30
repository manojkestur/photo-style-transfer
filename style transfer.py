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
