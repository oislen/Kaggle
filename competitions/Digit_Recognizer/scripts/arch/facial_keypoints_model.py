# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 19:47:52 2021

@author: oislen
"""

# import relevant libraries
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

def facial_keypoints_model(input_size, 
                           output_size
                           ):
    
    """
    """
        
    # set sequential model
    model = Sequential()
    
    # Define here your model
    act_func = 'relu'
    
    # set dropout rate
    drop_out_rate = 0.25
    
    # set kernel_size 
    kernel_size = (3, 3)
    
    # set pooling size
    pool_size = (2, 2)
    
    #-- conv layer 1 --#
    
    # create convonluational layer
    conv2d_layer_1 = Conv2D(32, 
                            kernel_size = kernel_size, 
                            activation = act_func, 
                            input_shape = input_size
                           )
    
    # add layer to model
    model.add(conv2d_layer_1)
    
    #-- conv layer 2 --#
    
    # create convonluational layer
    conv2d_layer_2 = Conv2D(64, 
                            kernel_size = kernel_size, 
                            activation = act_func
                           )
    
    # add layer to model
    model.add(conv2d_layer_2)
    
    #-- Max Pooling --#
    
    # create pooling layer
    max_pooling_layer = MaxPooling2D(pool_size = pool_size)
    
    # add layer to model
    model.add(max_pooling_layer)
    
    #-- flatten layer --#
    
    # create flatten layer
    flatten_layer = Flatten()
    
    # add layer to model
    model.add(flatten_layer)
    
    #-- dense layer 1 + dropout --#
    
    # create dense layer 
    dense_layer_1 = Dense(128, 
                          activation = act_func
                         )
    
    # add to model
    model.add(dense_layer_1)
    
    # create drop out layer
    drop_out_layer = Dropout(rate = drop_out_rate)
    
    # add to model
    model.add(drop_out_layer)
    
    #-- dense layer 2 + dropout --#
    
    # create dense layer
    dense_layer_2 = Dense(256, 
                          activation = act_func
                         )
    
    # add to model
    model.add(dense_layer_2)
    
    # create drop out layer
    drop_out_layer = Dropout(rate = drop_out_rate)
    
    # add to model
    model.add(drop_out_layer)
    
    #-- dense layer 3 --#
    
    # create final dense layer
    dense_layer_3 = Dense(output_size, 
                          activation = 'linear'
                         )
    
    # add to model
    model.add(dense_layer_3)
    
    return model