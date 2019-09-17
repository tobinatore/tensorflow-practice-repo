## tensorflow-practice-repo    
A repository containing every project I created / will create to practice using tensorflow and keras.    

# Prerequisites

To run the scripts found in this repository, you'll need to install the following Python packages:

 - sklearn
 - matplotlib
 - numpy
 - pandas
 - tensorflow
 - pillow
    

## Currently implemented:

 1. [MNIST fashion classification](#mnist)
 2. [CIFAR10 image classification](#cifar)
 3. [Stock prediction](#stocks)

   <a id="mnist"></a>
## MNIST fashion classification

Often seen as the new "Hello World" of neural networks,  classifying the data in the MNIST fashion dataset does not require a complicated neural network.
I decided to use a simple 3-layer architecture, consisting of an input layer with 784 neurons, connected to a dense hidden layer with 128 neurons and the output layer, which has 10 neurons. The hidden layer uses relu as activation function, whereas the output layer uses softmax to calculate the confidence rating.
After testing the model, an overview of the first 25 samples from the test set is displayed. Next to each image is a confidence chart, that shows how confident the model was when choosing the label.    
![An image showing different fashion items plus the label the model predicted.](https://i.gyazo.com/540beb1da232dfb304729142afe82155.png)    
As you can see, the model classified 22 out of 25 shown images correctly.
    
<a id="cifar"></a>  
## CIFAR10 image classification
    
The CIFAR10 dataset is a dataset consisting of 60,000 32x32 color images containing one of 10 object classes, with 6000 images per class. It is commonly used to train machine learning and computer vision algorithms.
To effectively classify the images in the dataset and any images the user chooses, I implemented a relatively simple convolutional neural network (CNN).    
My model consists of the following layers:
- an input layer (shape 32x32x3)
- a BatchNormalization layer
- a Conv2D layer followed by a MaxPool2D layer (3 times = 6 layers)
- a layer for flattening the input (keras.layers.Flatten)
- a dense layer with 512 neurons
- the output layer (dense) with 10 neurons    

The script creating the model will save it in the same directory it resides in. You can then use the classification script to classify any image you like.    

The model should have an accuracy somewhere in the high 70% range after 50 epochs. I'll try and optimize it when I've got a bit more time on my hands.

<a id="stocks"></a>  
## Stock prediction

An algorithm correctly and accurately prediction stock value has yet to be found, but predicting tendencies using a Long Short Term Memory network (LSTM) is already feasible and fairly easy.    
To tackle that problem I am using a model with 9 layers. These layers (except for the output layer) are always an LSTM layer with 50 neurons followed by a dropout layer. This is to avoid overfitting. A dense layer with 1 neuron functions as the output layer.    
After training and saving the model, it is used to predict the value of the stock. This prediction will then be overlaid over the actual data, to get a feeling of the accuracy of the prediction.    
Here an example using Alphabet:    
![A chart showing the predicted values.](https://i.gyazo.com/2e750889462ab03ca315bdc5c422f03f.png)    
And one for Microsoft:
![A chart showing the predicted values.](https://i.gyazo.com/8297d9a161f9c6e87164c8ace6a26c55.png)        
As you can see, the prediction captures the general tendencies pretty well.
