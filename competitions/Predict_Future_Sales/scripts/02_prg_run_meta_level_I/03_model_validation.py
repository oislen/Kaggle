# -*- coding: utf-8 -*-
"""
Created on Sun May 17 12:27:58 2020

@author: oislen
"""

import pandas as pd
from reference.calc_rmse import calc_rmse
from reference.plot_preds_vs_true import plot_preds_vs_true
from reference.plot_preds_hist import plot_preds_hist

def model_validation(pred_paths, valid_output_paths, model_name):
    
    """
    
    Model Validation Documentation
    
    """
    
    # extract out the validation output paths
    preds_valid_rmse = valid_output_paths['preds_valid_rmse']
    preds_test_rmse = valid_output_paths['preds_test_rmse']
    preds_vs_true_valid = valid_output_paths['preds_vs_true_valid']
    preds_vs_true_test = valid_output_paths['preds_vs_true_test']
    true_hist_valid = valid_output_paths['true_hist_valid']
    true_hist_test = valid_output_paths['true_hist_test']
    preds_hist_valid = valid_output_paths['preds_hist_valid']
    preds_hist_test = valid_output_paths['preds_hist_test']
    preds_hist_holdout = valid_output_paths['preds_hist_holdout']

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
    valid_rmse = calc_rmse(dataset = y_valid, tar = 'item_cnt_day', pred = 'y_valid_pred', out_fpath = preds_valid_rmse)
    test_rmse = calc_rmse(dataset = y_test, tar = 'item_cnt_day', pred = 'y_test_pred', out_fpath = preds_test_rmse)
    
    print('Validation Set RMSE:', valid_rmse)
    print('Test Set RMSE:', test_rmse)

    #-- Preds vs True --#
    
    print('Plotting predictions vs true tagret scatterplots ...')
    
    # create a scatterplot of predictions vs true
    plot_preds_vs_true(dataset = y_valid, tar = 'item_cnt_day', pred = 'y_valid_pred', model_name = model_name, out_fpath = preds_vs_true_valid)
    plot_preds_vs_true(dataset = y_test, tar = 'item_cnt_day', pred = 'y_test_pred', model_name = model_name, out_fpath = preds_vs_true_test)

    #-- Preds Hist --#
    
    print('Plotting predictions histograms ...')
    
    # create a hist of pred distribution
    plot_preds_hist(dataset = y_valid, pred = 'item_cnt_day', bins = 100, kde = False, model_name = model_name, out_fpath = true_hist_valid)
    plot_preds_hist(dataset = y_test, pred = 'item_cnt_day', bins = 100, kde = False, model_name = model_name, out_fpath = true_hist_test)
    plot_preds_hist(dataset = y_valid, pred = 'y_valid_pred', bins = 100, kde = False, model_name = model_name, out_fpath = preds_hist_valid)
    plot_preds_hist(dataset = y_test, pred = 'y_test_pred', bins = 100, kde = False, model_name = model_name, out_fpath = preds_hist_test)
    plot_preds_hist(dataset = y_holdout, pred = 'y_holdout_pred', bins = 100, kde = False, model_name = model_name, out_fpath = preds_hist_holdout)
    
    return 0
