# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 15:49:47 2021

@author: oislen
"""

from keras.models import Model
from keras.layers import Input, Dropout
from keras.layers import Conv2D, MaxPooling2D

def FCNN_Model(image_shape, n_targets):
    
    """
    
    After model compilation input size cannot be changed.
    
    So, we need create a function to have ability to change size later.
    
    """
    
    # set model constants
    kernel_size_1 = (5, 5)    
    kernel_size_2 = (8, 8)  
    kernel_size_3 = (1, 1)
    act_func = 'relu'
    pad = 'same'
    pool_size = (2, 2)
    drop_rate = 0.25
    
    inputs = Input(image_shape)
    
    # convolution layer 1
    conv_layer_1 = Conv2D(32, 
                         kernel_size = kernel_size_1, 
                         activation = act_func, 
                         padding = pad
                         )(inputs)
    
    # first pooling level
    pool_layer_1 = MaxPooling2D(pool_size)(conv_layer_1)
    
    # convoluation layer 2
    conv_layer_2 = Conv2D(32, 
                          kernel_size = kernel_size_1, 
                          activation = act_func, 
                          padding = pad
                          )(pool_layer_1)
    
    # second pooling level
    pool_layer_2 = MaxPooling2D(pool_size)(conv_layer_2)
    
    # third convolution layer
    con_layer_3 = Conv2D(128, 
                         kernel_size = kernel_size_2, 
                         activation = act_func
                        )(pool_layer_2)
    
    # drop out layer
    drop_layer_1 = Dropout(drop_rate)(con_layer_3)
    
    # fourth convoluation layer
    conv_layer_4 = Conv2D(64, 
                          kernel_size = kernel_size_3, 
                          activation = act_func
                          )(drop_layer_1)
    
    # drop output later
    drop_layer_2 = Dropout(drop_rate)(conv_layer_4)

    # create prediction layer
    conv_layer_5 = Conv2D(2, 
                          kernel_size_3, 
                          activation='linear'
                          )(drop_layer_2)
    
    # create final model
    model = Model(inputs = inputs, 
                  outputs = conv_layer_5
                 )
    
    return model