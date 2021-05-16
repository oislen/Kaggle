# -*- coding: utf-8 -*-
"""
Created on Sat May 15 08:34:58 2021

@author: oislen
"""

import numpy as np
import pandas as pd

def calc_rmse(dataset, 
              tar, 
              pred, 
              out_fpath = None
              ):
    
    """
    
    Calculate RMSE Documenation
    
    Function Overview
    
    This function calculates roost mean square error for a given dataset, target variable and predictor variable.
    The results can also be written to a specified file path as a .csv file.
    
    Defaults
    
    calc_rmse(dataset, 
              tar, 
              pred, 
              out_fpath = None
              )
    
    Parameters
    
    dataset - DataFrame, the data to calculate RMSE for
    tar - String, the target column name of the given dataset
    pred - String, the predictions column name of the given dataset
    out_fpath - String, the file path to output the results as a .csv file, default is None
    
    Returns
    
    rmse_df - DataFrame, the calculated RMSE of the given dataset
    
    Example
    
    calc_rmse(dataset = y_valid, 
              tar = 'item_cnt_day', 
              pred = 'y_valid_pred',
              out_fpath = preds_valid_rmse
              )
    
    """
    
    # take a depp copy of the data
    data = dataset.copy(True)
    
    # calculate RMSE
    rmse = np.sqrt(((data[tar] - data[pred]) ** 2).sum() / data.shape[0])
    
    # convert result to a dictionary
    rmse_dict = {'RMSE':[rmse]}
    
    # convert dictionary into dataframe
    rmse_df = pd.DataFrame(rmse_dict, index = [0])
    
    # if outputing results
    if out_fpath != None:
        
        # write to speficied file path
        rmse_df.to_csv(out_fpath, 
                       index = False
                       )
        
    return rmse_df