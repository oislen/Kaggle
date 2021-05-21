# -*- coding: utf-8 -*-
"""
Created on Sun May 17 12:29:39 2020

@author: oislen
"""

import cons
import pandas as pd

def format_kaggle_preds(model_name):
    
    """
    
    Format Kaggle Predictions Documentation
    
    Function Overview
    
    This functions formats and prepares the kaggle competition predicitons for submission to the kaggle competition platform
    
    Defaults
    
    format_kaggle_preds(model_name)
    
    Paraemeters
    
    model_name - String, the name of the model
    
    Returns
    
    0 for successful execution
    
    Example
    
    format_kaggle_preds(model_name = model_name)
    
    """
    
    # extract out the prediciton paths
    y_holdout_preds_path = cons.result_output_paths['y_holdout_preds_path'].format(model_type = model_name)
    kaggle_preds = cons.result_output_paths['kaggle_preds'].format(model_type = model_name)
    
    # load in holdout predictions
    y_holdout = pd.read_feather(y_holdout_preds_path)
    
    # extract out test predictions
    holdout_subset_filt = y_holdout['holdout_subset_ind'] == 1
    holdout_out = y_holdout.loc[holdout_subset_filt, ['ID', 'y_holdout_pred']]
    holdout_out = holdout_out.rename(columns = {'y_holdout_pred':'item_cnt_month'})
    holdout_out_sort = holdout_out.sort_values(by = ['ID']).astype(int)
    
    # output predictions as csv file
    holdout_out_sort.to_csv(kaggle_preds,
                            index = False
                            )
    
    return 0