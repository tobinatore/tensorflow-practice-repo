import matplotlib.pyplot as plt
from tensorflow import keras
from PIL import Image
import numpy as np
import argparse
import json
import h5py

# -------------------------------------------------------------------------------------------- #
#   A script for using a pretrained Convolutional Neural Network to classify images into one   #
#                                 of ten categories.                                           #
#                                                                                              #
#                                Dataset used for this task:                                   #
#                           CIFAR10, imported directly from keras                              #
# -------------------------------------------------------------------------------------------- #

CLASS_NAMES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


def fix_layer0(filename, input_shape, dtype):
    """
    There is a known bug in tensorflow v.1.14, where models with an input layer can't be loaded.
    This function fixes the wrongly exported layer.
    :param filename: String
    :param input_shape: 3-tuple (int, int, int)
    :param dtype: String
    """
    with h5py.File(filename, 'r+') as f:
        model_config = json.loads(f.attrs['model_config'].decode('utf-8'))
        layer0 = model_config['config']['layers'][0]['config']
        layer0['input_shape'] = input_shape
        layer0['dtype'] = dtype
        f.attrs['model_config'] = json.dumps(model_config).encode('utf-8')


parser = argparse.ArgumentParser(description='Classify images using a CNN trained on CIFAR10')
parser.add_argument('img', type=str, help='Path to image')
parser.add_argument('model', type=str, help='Path to pretrained model')

args = parser.parse_args()

fix_layer0(args.model, (32, 32, 3), 'float32')

img = Image.open(args.img)

img = img.resize((32, 32), Image.ANTIALIAS)
img = np.array(img, dtype="float32")
img /= 255
img = img.reshape(-1, 32, 32, 3)

model = keras.models.load_model(args.model)
prediction = model.predict(img)
max_confidence = np.argmax(prediction)

plt.title("Prediction: " + CLASS_NAMES[max_confidence])
plt.imshow(img[0].reshape(32, 32, 3))
plt.show()
