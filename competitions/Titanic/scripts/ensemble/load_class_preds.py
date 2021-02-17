# -*- coding: utf-8 -*-
"""
Created on Sat Feb 13 10:28:48 2021

@author: oislen
"""

# import relevant libraries
import pandas as pd
import cons

def load_class_preds(model_keys,
                     pred_data_fpath, 
                     join_col = 'PassengerId'
                     ):
    
    """
    
    Load Classification Prediction Documentation
    
    Function Overview
    
    This function loads the classification predictions of the test for each of the best single sklearn models
    
    Defaults
    
    load_class_preds(model_keys,
                     pred_data_fpath, 
                     join_col = 'PassengerId'
                     )
    
    Parameters
    
    model_keys - Dictionary Keys, the model keys from the model parameters dictionary, see cons.py
    pred_data_fpath - String, the file path to the model predictions
    join_col - String, the column to join the model predictions on, default is 'PassengerId'
    
    Returns
    
    preds_df - DataFrame, the join model predictions for all best single sklearn classifiers
    
    Example
    
    load_class_preds(model_keys = ['gbc', 'rfc', 'abc', 'etc', 'svc'],
                     pred_data_fpath = 'C:\\Users\\...\\preds\\{}_preds.csv', 
                     join_col = 'PassengerId'
                     )
        
    """
    
    # set the predictor column
    pred_col = 'Survived'

    # loop through the model keys
    for idx, key in enumerate(model_keys):
        
        model_pred_data_fpath = pred_data_fpath.format(key)
        
        # load in the model predicstons
        model_preds_df = pd.read_csv(model_pred_data_fpath, sep = cons.sep)
        
        # create the new predictions column name
        new_pred_col = '{}_{}'.format(key, pred_col)
        
        # rename the predictions column
        model_preds_df = model_preds_df.rename(columns = {pred_col:new_pred_col})
        
        if idx == 0:
            
            preds_df = model_preds_df
            
        else:
            
            # join to the output dataframe
            preds_df = pd.merge(left = preds_df, right = model_preds_df, on = join_col)
    
    return preds_df