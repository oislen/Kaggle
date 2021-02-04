# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 15:49:47 2021

@author: oislen
"""

# load in relevant libraries
from keras.models import Model
from keras.layers import Input, Dropout
from keras.layers import Conv2D, MaxPooling2D

def FCNN_Model(image_shape, 
               n_targets
               ):
    
    """
    
    Fully Convolution Nueral Network (FCC) Model Documentation
    
    Function Overview
    
    This function generates a Fully Convolution Nueral Network Model architecture:
        
        1. Conv2D 
            - filters: 32  
            - kernal: 5 x 5  
            - activation: relu 
             - padding: same
            
        2. MaxPooling2D 
            - pool: 2 x 2 
            
        3. Conv2D 
            - filters: 32  
            - kernal: 5 x 5 
            - activation: relu 
             - padding: same
            
        4. MaxPooling2D 
            - pool: 2 x 2 
            
        5. Conv2D 
            - filters: 128  
            - kernal: 7 x 7 
            - activation: relu 
             - padding: same
            
        6. Dropout 
            - rate: 0.25 
            
        7. Conv2D 
            - filters: 64  
            - kernal: 1 x 1 
            - activation: relu 
             - padding: same
            
        8. Dropout 
            - rate: 0.25 
            
        9. Conv2D 
            - nodes: n target  
            - kernal: 1 x 1  
            - activation: linear 
             - padding: same
        
    Defaults
    
    FCNN_Model(image_shape, 
               n_targets
               )
    
    Parameters
    
    image_shape - the input image shape / dimensions
    n_targets - the number of target classes
    
    Returns
    
    model - keras.Model, the FCNN model
    
    Example
    
    FCNN_Model(image_shape = (28, 28, 1), 
               n_targets = 2
               )
    
    """
    
    # set model constants
    kernel_size_1 = (5, 5)    
    kernel_size_2 = (7, 7)  
    kernel_size_3 = (1, 1)
    act_func = 'relu'
    pad = 'same'
    pool_size = (2, 2)
    drop_rate = 0.25
    
    inputs = Input(image_shape)
    
    # convolution layer 1
    conv_layer_1 = Conv2D(filters = 32, 
                         kernel_size = kernel_size_1, 
                         activation = act_func, 
                         padding = pad
                         )(inputs)
    
    # first pooling level
    pool_layer_1 = MaxPooling2D(pool_size)(conv_layer_1)
    
    # convoluation layer 2
    conv_layer_2 = Conv2D(filters = 32, 
                          kernel_size = kernel_size_1, 
                          activation = act_func, 
                          padding = pad
                          )(pool_layer_1)
    
    # second pooling level
    pool_layer_2 = MaxPooling2D(pool_size)(conv_layer_2)
    
    # third convolution layer
    conv_layer_3 = Conv2D(filters = 128, 
                          kernel_size = kernel_size_2, 
                          activation = act_func, 
                          padding = pad
                         )(pool_layer_2)

    # drop out layer
    drop_layer_1 = Dropout(drop_rate)(conv_layer_3)
    
    # fourth convoluation layer
    conv_layer_4 = Conv2D(filters = 64, 
                          kernel_size = kernel_size_3, 
                          activation = act_func, 
                          padding = pad
                          )(drop_layer_1)
    
    # drop output later
    drop_layer_2 = Dropout(drop_rate)(conv_layer_4)

    # create prediction layer
    conv_layer_5 = Conv2D(filters = n_targets, 
                          kernel_size = kernel_size_3, 
                          activation='linear', 
                          padding = pad
                          )(drop_layer_2)

    # create final model
    model = Model(inputs = inputs, 
                  outputs = conv_layer_5
                 )
    
    return model
