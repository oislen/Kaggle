# -*- coding: utf-8 -*-
"""
Created on Sun May 17 12:27:58 2020

@author: oislen
"""

import pandas as pd
import seaborn as sns
import numpy as np

def model_validation(pred_paths):
    
    """
    """
    
    # extract out the prediciton paths
    y_valid_preds_path = pred_paths['y_valid_preds_path']
    y_test_preds_path = pred_paths['y_test_preds_path']
    y_holdout_preds_path = pred_paths['y_holdout_preds_path']
    
    # load in the predictions
    y_holdout = pd.read_csv(y_holdout_preds_path)
    y_test = pd.read_csv(y_test_preds_path) 
    
    # TODO: incorporate a whole script in model evaluation here
    y_test['y_test_pred'].value_counts()
    y_holdout['y_holdout_pred'].value_counts()
    
    # create confusion matrix
    #pd.crosstab(index = y_valid['item_cnt_day'], 
    #            columns = y_valid['y_valid_pred']
    #            )
    
    # create confusion matrix
    sns.scatterplot(x = 'item_cnt_day', y = 'y_test_pred', data = y_test)
    
    # create a hist of pred distribution
    sns.distplot(a = y_test['item_cnt_day'], bins = 100, kde = False)
    sns.distplot(a = y_test['y_test_pred'], bins = 100, kde = False)
    
    # create a hist of pred distribution
    sns.distplot(a = y_holdout['y_holdout_pred'], bins = 100, kde = False)
    
    # calculate RMSE
    np.sqrt(((y_test['item_cnt_day'] - y_test['y_test_pred']) ** 2).sum() / y_test.shape[0])
