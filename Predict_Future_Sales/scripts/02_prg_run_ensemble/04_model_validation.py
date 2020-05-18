# -*- coding: utf-8 -*-
"""
Created on Sun May 17 12:27:58 2020

@author: oislen
"""

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

def model_validation(pred_paths):
    
    """
    """
    
    print('Loading model predictions ...')
    
    # extract out the prediciton paths
    y_valid_preds_path = pred_paths['y_valid_preds_path']
    y_test_preds_path = pred_paths['y_test_preds_path']
    y_holdout_preds_path = pred_paths['y_holdout_preds_path']
    
    # load in the predictions
    y_valid = pd.read_csv(y_valid_preds_path)
    y_test = pd.read_csv(y_test_preds_path) 
    y_holdout = pd.read_csv(y_holdout_preds_path)
    
    # prediction value counts
    print('Validation Predictions:', y_valid['y_valid_pred'].value_counts())
    print('Test Predictions:', y_test['y_test_pred'].value_counts())
    print('Holdout Predictions:', y_holdout['y_holdout_pred'].value_counts())
    
    #-- RMSE --#
    
    # calculate RMSE
    valid_rmse = np.sqrt(((y_valid['item_cnt_day'] - y_valid['y_valid_pred']) ** 2).sum() / y_valid.shape[0])
    print('Validation Set RMSE:', valid_rmse)
    
    # calculate RMSE
    test_rmse = np.sqrt(((y_test['item_cnt_day'] - y_test['y_test_pred']) ** 2).sum() / y_test.shape[0])
    print('Test Set RMSE:', test_rmse)
    
    #-- Cross Tab --#
    
    # create confusion matrix
    valid_tab = pd.crosstab(index = y_valid['item_cnt_day'], 
                            columns = y_valid['y_valid_pred']
                            )
    print(valid_tab)
    
    # create confusion matrix
    test_tab = pd.crosstab(index = y_test['item_cnt_day'], 
                           columns = y_test['y_valid_pred']
                           )
    print(test_tab)
    
    #-- Preds vs True --#
    
    # create confusion matrix
    sns.scatterplot(x = 'item_cnt_day', y = 'y_valid_pred', data = y_valid)
    plt.show() 
    
    # create confusion matrix
    sns.scatterplot(x = 'item_cnt_day', y = 'y_test_pred', data = y_test)
    plt.show() 
    
    #-- Pred Hist --#
    
    # create a hist of pred distribution
    sns.distplot(a = y_test['item_cnt_day'], bins = 100, kde = False)
    plt.show() 
    
    sns.distplot(a = y_test['y_test_pred'], bins = 100, kde = False)
    plt.show() 
        
    # create a hist of pred distribution
    sns.distplot(a = y_holdout['y_holdout_pred'], bins = 100, kde = False)
    plt.show() 
    
    return
