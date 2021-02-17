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
    
    Load Classification Models Documentation
    
    Function Overview
    
    This function loads in all of the best saved sklearn single classifier models.
    
    Defaults
    
    load_class_models(model_keys, 
                      model_fpath
                      )
    
    Parametes
    
    model_keys, Dictionary Keys, the model keys from the model parameters dictionary, see cons.py
    model_fpath, String, the input file path to the sklearn models
    
    Returns
    
    class_models_dict - Dictionary, the single sklearn model classifiers
    
    Example
    
    load_class_models(model_keys = ['gbc', 'rfc', 'abc', 'etc', 'svc'], 
                      model_fpath = 'C:\\Users\\...\\{}_best_model.pkl'
                      )
    
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