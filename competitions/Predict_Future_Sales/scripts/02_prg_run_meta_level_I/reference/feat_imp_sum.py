# -*- coding: utf-8 -*-
"""
Created on Sat May 15 08:32:18 2021

@author: oislen
"""

import pandas as pd

def feat_imp_sum(model, 
                 pred_cols, 
                 feat_imp_fpath
                 ):
    
    """
    
    Feature Importance Summary Documentation
    
    Function Overview
    
    This functoin creates a feature importance summary for a given fitted model and set of predictor columns. 
    The results are outputed to the specified file path as a .csv file
    
    Defaults
    
    feat_imp_sum(model, 
                 pred_cols, 
                 feat_imp_fpath
                 )
    
    Parameters
    
    model - Sklearn Model, the fitted sklearn model to extract the feature importance results from
    pred_cols- List of Strings, the predictor columns used in the fitted sklearn model
    feat_imp_fpath - String, the output file path to save the feature importance results as a .csv file
    
    Returns
    
    feat_imp - DataFrame, the feature importance results of the fitted sklearn model
    
    Example
    
    feat_imp_sum(model =  gbr, 
                 pred_cols = pred_cols, 
                 feat_imp_fpath = feat_imp_fpath
                 )
    
    """
    
    # extract feature importance
    feat_imp = pd.DataFrame({'attr':pred_cols,
                             'feat_imp':model.feature_importances_ * 100
                             })
    
    # sort by importance
    feat_imp = feat_imp.sort_values(by = 'feat_imp', ascending = False)
    
    # reset the index
    feat_imp = feat_imp.reset_index(drop = True)
    
    # output to .csv file
    feat_imp.to_csv(feat_imp_fpath, index = False)
    
    return feat_imp
    