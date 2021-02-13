# -*- coding: utf-8 -*-
"""
Created on Sat Feb 13 10:39:57 2021

@author: oislen
"""

# import relevant libraries
import joblib

def load_class_models(model_keys, 
                      model_fpath
                      ):
    
    """
    """
    
    # create a dictionary to hold the class models
    class_models_dict = {}
    
    # loop through the model keys
    for key in model_keys:
        
        # create the full class model file path
        class_model_fpath = model_fpath.format(key)
        
        # load in the classification model 
        class_model = joblib.load(class_model_fpath)
        
        # assign to the output dictionary
        class_models_dict[key] = class_model
        
    return class_models_dict