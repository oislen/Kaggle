# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 12:40:29 2021

@author: oislen
"""

# import relevant libraries
import pandas as pd

def comp_valid_perf_metrics(model_keys, 
                            perf_metrics_fpath,
                            use_cols = ['Acc', 'Prec', 'Recall', 'AUC', 'F1']
                            ):
    
    """
    
    Compare Validation Performance Metrics Documentation
    
    Function Overview
    
    This function loads and compares all the validation performance metrics for each of the single sklearn classifer models.
    
    Defaults
    
    comp_valid_perf_metrics(model_keys, 
                            perf_metrics_fpath,
                            use_cols = ['Acc', 'Prec', 'Recall', 'AUC', 'F1']
                            )
    
    Parameters
    
    model_keys - Dictionary Keys, the model keys from the model parameters dictionary, see cons.py
    perf_metrics_fpath - String, the file path to the model predictions
    user_cols - List of Strings, the model performance metrics to load in and compare for each model, default is ['Acc', 'Prec', 'Recall', 'AUC', 'F1']
    
    Returns
    
    perf_metrics - DataFrame, the performance metrics comparison report
    
    Example
    
    comp_valid_perf_metrics(model_keys = ['gbc', 'rfc', 'abc', 'etc', 'svc'], 
                            perf_metrics_fpath = 'C:\\Users\\...\\model_results\\{}\\{}_perf_metrics.csv',
                            use_cols = ['Acc', 'Prec', 'Recall', 'AUC', 'F1']
                            )
    
    """
    
    # create empty dataframe to hold the results
    perf_metrics = pd.DataFrame()
    
    # iterate over the different models
    for model in model_keys:
        
        # generate the performance metrics file path
        model_results_fpath = perf_metrics_fpath.format(model, model)
        
        # load in model performance metrics
        model_perf_metrics = pd.read_csv(model_results_fpath, sep = '|', usecols = use_cols)
        
        # rename frame index
        model_perf_metrics = model_perf_metrics.rename(index = {0:model})
        
        # concatenate dataframe objects
        perf_metrics = pd.concat(objs = [perf_metrics, model_perf_metrics], axis = 0)
        
    # order results by accuracy
    perf_metrics = perf_metrics.sort_values(by = ['Acc'], ascending = False)
    
    return perf_metrics
        
        
        