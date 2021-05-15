# -*- coding: utf-8 -*-
"""
Created on Sat May 15 08:34:58 2021

@author: oislen
"""

import numpy as np

def format_preds(dataset, 
                 preds_cols
                 ):
    
    """
    
    Format Predictions Documentation
    
    Function Overvieew
    
    This function formats the predictions by rounding the down the final predictions to the nearest values
    
    Defaults
    
    format_preds(dataset, 
                 preds_cols
                 )
    
    Parameters
    
    dataset - DataFrame, the final predictins to round down to the nearest values
    preds_cols - Strings, the predictions column name
    
    Returns
    
    data - DataFrame, the formated final predictions
    
    Example
    
    format_preds(dataset = y_test, 
                 preds_cols = 'y_test_pred'
                 )
    
    """
    
    # take a deep copy of the data
    data = dataset.copy(True)
    
    # map items with no historical sell to 0
    no_sales_hist_filt = data['no_sales_hist_ind'] == 1
    data.loc[no_sales_hist_filt, [preds_cols]] = 0
    
    # round down remaining results to nearest value
    data[preds_cols] = data[preds_cols].apply(lambda x: np.floor(x))
    print(data[preds_cols].value_counts())
    
    return data