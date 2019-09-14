import numpy as np
from tensorflow import keras

# -------------------------------------------------------------------------------------------- #
# A script for training a Convolutional Neural Network to classify images into ten categories. #
#                                                                                              #
#                                Dataset used for this task:                                   #
#                           CIFAR10, imported directly from keras                              #
# -------------------------------------------------------------------------------------------- #

# Declaring the constants
#   BATCH_SIZE:     no. of samples to work through before updating model parameters
#   NUM_CLASSES:    number of classes
#   EPOCHS:         how often the model is trained on the data
BATCH_SIZE = 32
NUM_CLASSES = 10
EPOCHS = 50

# Loading the data
cifar10 = keras.datasets.cifar10
(images_train, labels_train), (images_test, labels_test) = cifar10.load_data()

# Preprocessing the images.
# Converting the images to arrays and normalizing them
images_train = np.array(images_train, dtype="float32")
images_test = np.array(images_test, dtype="float32")
images_train /= 255
images_test /= 255

# Using one-hot-encoding for the labels
labels_train = keras.utils.to_categorical(labels_train, NUM_CLASSES)
labels_test = keras.utils.to_categorical(labels_test, NUM_CLASSES)

# Defining the model.
model = keras.Sequential()
model.add(keras.layers.InputLayer(input_shape=(32, 32, 3)))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Conv2D(32, (3, 3), padding="same", input_shape=(32, 32, 3), activation="relu"))
model.add(keras.layers.MaxPool2D(pool_size=(2, 2)))
model.add(keras.layers.Conv2D(64, (3, 3), padding="same"))
model.add(keras.layers.MaxPool2D(pool_size=(2, 2)))
model.add(keras.layers.Conv2D(64, (3, 3), padding="same"))
model.add(keras.layers.MaxPool2D(pool_size=(2, 2)))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(512, activation="relu"))
model.add(keras.layers.Dense(NUM_CLASSES, activation="softmax"))

# Compiling and training the model
model.compile(loss="categorical_crossentropy", optimizer="adadelta", metrics=["acc"])
model.fit(images_train, labels_train, batch_size=BATCH_SIZE, epochs=EPOCHS)

# Evaluating and saving the model for later use.
scores = model.evaluate(images_train, labels_train)

print("--- Scores ---")
print("Loss: ", scores[0])
print("Accuracy: ", scores[1])

model.save("cifar10_model.h5")
