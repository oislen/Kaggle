# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 12:40:29 2021

@author: oislen
"""


# import relevant libraries
import pandas as pd
import os
import cons

def comp_valid_perf_metrics(model_keys, 
                            perf_metrics_fpath
                            ):
    
    """
    """
    
    # create empty dataframe to hold the results
    perf_metrics = pd.DataFrame()
    
    # iterate over the different models
    for model in model_keys:
        
        # generate the performance metrics file path
        model_results_fpath = perf_metrics_fpath.format(model, model)
        
        # load in model performance metrics
        use_cols = ['Acc', 'Prec', 'Recall', 'AUC', 'F1']
        model_perf_metrics = pd.read_csv(model_results_fpath, sep ='|', usecols = use_cols)
        
        # rename frame index
        model_perf_metrics = model_perf_metrics.rename(index = {0:model})
        
        # concatenate dataframe objects
        perf_metrics = pd.concat(objs = [perf_metrics, model_perf_metrics], axis = 0)
        
    # order results by accuracy
    perf_metrics = perf_metrics.sort_values(by = ['Acc'], ascending = False)
    
    return perf_metrics
        
        
        