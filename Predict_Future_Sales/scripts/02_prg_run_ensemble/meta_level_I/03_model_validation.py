# -*- coding: utf-8 -*-
"""
Created on Sun May 17 12:27:58 2020

@author: oislen
"""

import pandas as pd
import utilities_ensemble as utl_ens

def model_validation(pred_paths):
    
    """
    
    Model Validation Documentation
    
    """
    
    print('Loading model predictions ...')
    
    # extract out the prediciton paths
    y_valid_preds_path = pred_paths['y_valid_preds_path']
    y_test_preds_path = pred_paths['y_test_preds_path']
    y_holdout_preds_path = pred_paths['y_holdout_preds_path']
    
    # load in the predictions
    y_valid = pd.read_feather(y_valid_preds_path)
    y_test = pd.read_feather(y_test_preds_path) 
    y_holdout = pd.read_feather(y_holdout_preds_path)
    
    # prediction value counts
    print('Validation Predictions:')
    print(y_valid['y_valid_pred'].value_counts())
    print('Test Predictions:')
    print(y_test['y_test_pred'].value_counts())
    print('Holdout Predictions:')
    print(y_holdout['y_holdout_pred'].value_counts())
    
    #-- RMSE --#
    
    print('Calculating RMSE ...')
    
    # calculate RMSE
    valid_rmse = utl_ens.calc_rmse(dataset = y_valid, tar = 'item_cnt_day', pred = 'y_valid_pred', out_fpath = None)
    test_rmse = utl_ens.calc_rmse(dataset = y_test, tar = 'item_cnt_day', pred = 'y_test_pred', out_fpath = None)
    
    print('Validation Set RMSE:', valid_rmse)
    print('Test Set RMSE:', test_rmse)

    #-- Preds vs True --#
    
    print('Plotting predictions vs true tagret scatterplots ...')
    
    # create a scatterplot of predictions vs true
    utl_ens.plot_preds_vs_true(dataset = y_valid, tar = 'item_cnt_day', pred = 'y_valid_pred', out_fpath = None)
    utl_ens.plot_preds_vs_true(dataset = y_test, tar = 'item_cnt_day', pred = 'y_test_pred', out_fpath = None)

    #-- Preds Hist --#
    
    print('Plotting predictions histograms ...')
    
    # create a hist of pred distribution
    utl_ens.plot_preds_hist(dataset = y_valid, pred = 'item_cnt_day', bins = 100, kde = False, out_fpath = None)
    utl_ens.plot_preds_hist(dataset = y_test, pred = 'item_cnt_day', bins = 100, kde = False, out_fpath = None)
    utl_ens.plot_preds_hist(dataset = y_valid, pred = 'y_valid_pred', bins = 100, kde = False, out_fpath = None)
    utl_ens.plot_preds_hist(dataset = y_test, pred = 'y_test_pred', bins = 100, kde = False, out_fpath = None)
    utl_ens.plot_preds_hist(dataset = y_holdout, pred = 'y_holdout_pred', bins = 100, kde = False, out_fpath = None)
    
    return
