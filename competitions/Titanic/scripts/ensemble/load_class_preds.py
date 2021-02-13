# -*- coding: utf-8 -*-
"""
Created on Sat Feb 13 10:28:48 2021

@author: oislen
"""

# import relevant libraries
import pandas as pd

def load_class_preds(model_keys, 
                     join_col,
                     pred_data_fpath
                     ):
    
    """
    """
    
    pred_col = 'Survived'

    # loop through the model keys
    for idx, key in enumerate(model_keys):
        
        model_pred_data_fpath = pred_data_fpath.format(key)
        
        # load in the model predicstons
        model_preds_df = pd.read_csv(model_pred_data_fpath)
        
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