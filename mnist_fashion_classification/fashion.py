import matplotlib.pyplot as plt
from tensorflow import keras
import numpy as np

# ------------------------------------------------------------------------------ #
# A neural network doing some basic classification on the MNIST fashion dataset. #
#       Classifies images of clothing in 10 categories (see CLASS_NAMES).        #
# ------------------------------------------------------------------------------ #


# class names for the number the neural network returns
CLASS_NAMES = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


def plot_image(i, predictions_array, true_label, img):
    """
    Plotting a single image from the test set with caption
    :param i: Integer
    :param predictions_array: numpy.nparray
    :param true_label: String
    :param img: numpy.nparray
    """
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(CLASS_NAMES[predicted_label],
                                         100 * np.max(predictions_array),
                                         CLASS_NAMES[true_label]),
               color=color)


def plot_value_array(i, predictions_array, true_label):
    """
    Function for printing the different confidence ratings.
    :param i: Integer
    :param predictions_array: numpy.nparray
    :param true_label: String
    """
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    plot[predicted_label].set_color('red')
    plot[true_label].set_color('blue')


# Specifying the used data set and loading the data
data = keras.datasets.fashion_mnist
(images_train, labels_train), (images_test, labels_test) = data.load_data()

# Normalizing the values to be between 0 and 1
images_train = images_train/255.0
images_test = images_test/255.0

# Defining and training the model, topology: 784-128-10
model = keras.Sequential([keras.layers.Flatten(input_shape=(28,28)),
                          keras.layers.Dense(128, activation="relu"),
                          keras.layers.Dense(10, activation="softmax")])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(images_train, labels_train, epochs=5)

# Classifying the images from the test set
prediction = model.predict(images_test)

# Plotting the first X test images, their predicted label, and the true label
# Coloring correct predictions in blue, incorrect predictions in red
num_rows = 5
num_cols = 5
num_images = num_rows*num_cols

plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2*num_cols, 2*i+1)
    plot_image(i, prediction, labels_test, images_test)
    plt.subplot(num_rows, 2*num_cols, 2*i+2)
    plot_value_array(i, prediction, labels_test)
plt.show()
