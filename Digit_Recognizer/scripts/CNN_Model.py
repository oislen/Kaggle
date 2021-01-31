# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 16:00:25 2021

@author: oislen
"""

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, Dropout
from keras.layers.advanced_activations import LeakyReLU

def CNN_Model(input_shape, n_targets):
    """
    cifar10 coursea assignment
    Define your model architecture here.
    Returns `Sequential` model.
    """
    model = Sequential()

    # convolution layer 1
    # kernel size (3, 3)
    # filter 16
    # leakyrelu activation
    model.add(Conv2D(filters = 16, kernel_size = (3, 3), padding = 'same', name = 'convolution_layer_1', input_shape=input_shape))
    model.add(LeakyReLU(0.1))
    
    # convolution layer 2
    # kernel size (3, 3)
    # filter 32
    # leakyrelu activation
    model.add(Conv2D(filters = 32, kernel_size = (3, 3), padding = 'same', name = 'convolution_layer_2'))
    model.add(LeakyReLU(0.1))
    
    # pooling layer 1
    # dropout of 0.25
    model.add(MaxPooling2D(pool_size = (2, 2), name = 'pooling_layer_1'))
    model.add(Dropout(0.25))
    
    # convolution layer 3
    # kernel size (3, 3)
    # filter 32
    # leakyrelu activation
    model.add(Conv2D(filters = 32, kernel_size = (3, 3), padding = 'same', name = 'convolution_layer_3'))
    model.add(LeakyReLU(0.1))
    
    # convolution layer 4
    # kernel size (3, 3)
    # filter 64
    # leakyrelu activation
    model.add(Conv2D(filters = 64, kernel_size = (3, 3), padding = 'same', name = 'convolution_layer_4'))
    model.add(LeakyReLU(0.1))
    
    # pooling layer 2
    # dropout of 0.25
    model.add(MaxPooling2D(pool_size = (2, 2), name = 'pooling_layer_2'))
    model.add(Dropout(0.25))
    
    # flatten layer
    model.add(Flatten())
    
    # dense layer 1
    # 256 neurons
    model.add(Dense(units = 256, name = 'dense_layer_1'))
    
    # dropout of 0.5
    model.add(Dropout(0.5))
    
    # dense layer 2
    # 10 neurons
    model.add(Dense(units = n_targets, name = 'dense_layer_2'))
    model.add(Activation("softmax"))
    
    return model