# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 15:22:57 2021

@author: oislen
"""

from keras.models import Model
from keras.layers import Flatten, Dense, Input, Dropout
from keras.layers import Conv2D, MaxPooling2D

def LeNet_Model(input_shape, n_targets):
    
    """
    """
    
    # set function constants
    act_func = 'relu'
    kernel_size = (5, 5)
    pad = 'same'
    pool_size = (2, 2)
    drop_rate = 0.25
    
    # Classification model
    # You can start from LeNet architecture
    inputs = Input(shape = input_shape)
    
    # first convulation layer
    conv_layer_1 = Conv2D(32, 
                          kernel_size = kernel_size, 
                          activation = act_func, 
                          padding = pad
                          )(inputs)
    
    # pooling layer
    pool_layer_1 = MaxPooling2D(pool_size)(conv_layer_1)
    
    # second convulation later
    conv_layer_2 = Conv2D(32, 
                          kernel_size = kernel_size, 
                          activation = act_func,
                          padding = pad
                          )(pool_layer_1)
    
    # pooling layer
    pool_layer_2 = MaxPooling2D(pool_size)(conv_layer_2)
    
    # flatten inputs
    flat_layer = Flatten()(pool_layer_2)
    
    # first dense layer
    dense_layer_1 = Dense(128, 
                          activation = act_func
                          )(flat_layer)
    
    # drop out layer
    drop_layer_1= Dropout(drop_rate)(dense_layer_1)
    
    # second dense layer
    dense_layer_2 = Dense(64, 
                          activation = act_func
                     )(drop_layer_1)
    
    # drop out layer
    drop_layer_2 = Dropout(drop_rate)(dense_layer_2)

    # set prediction layer
    dense_layer_3 = Dense(n_targets, 
                          activation = 'softmax'
                          )(drop_layer_2)
    
    # create level
    model = Model(inputs = inputs, 
                  outputs = dense_layer_3
                 )
    
    return model