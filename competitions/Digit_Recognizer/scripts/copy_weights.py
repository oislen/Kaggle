# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 14:58:41 2021

@author: oislen
"""

def copy_weights(base_model, 
                 target_model
                ):
    
    """
    
    Copy Weights Function
    
    Function Overview
    
    This function copies weights from a base keras model to a target keras model.
    Note the models musts have an equal number of parameters at each network layer see keras.model.summary()
    
    Defaults
    
    copy_weights(base_model, 
                 target_model
                )
    
    Parameters
    
    base_model, keras.model, the base model to copy weights from
    target_model, keras.model, the target model to copy weights to
    
    Returns
    
    0 for successful execution
    
    copy_weights(base_model = LeNet_Model, 
                 target_model = FCNN_Model
                )
    
    """
    
    # create list to hold new weights for fcnn
    new_fcnn_weights = []
    
    # extract weights from base model and fcnn
    prev_target_weights = target_model.get_weights()
    prev_base_weights = base_model.get_weights()
    
    # zip previous weights into dictionary
    prev_dict_weights = zip(prev_target_weights, prev_base_weights)
    
    # loop through conents of zip
    for prev_target_weight, prev_base_weight in prev_dict_weights:
        
        # append the new fccnn weights
        new_fcnn_weights.append(prev_base_weight.reshape(prev_target_weight.shape))
    
    # set the new fcnn weights
    target_model.set_weights(new_fcnn_weights)
    
    return 0