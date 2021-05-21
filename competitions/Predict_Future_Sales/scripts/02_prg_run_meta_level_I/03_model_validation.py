# -*- coding: utf-8 -*-
"""
Created on Sun May 17 12:27:58 2020

@author: oislen
"""

import cons
import pandas as pd
from calc_rmse import calc_rmse
from plot_preds_vs_true import plot_preds_vs_true
from plot_preds_hist import plot_preds_hist

def model_validation(model_name):
    
    """
    
    Model Validation Documentation
    
    Function Overview
    
    This function performs validation checks on the predictions for the validation, test set and hold out set.
    
    Defaults
    
    model_validation(mod_preds, 
                     model_name
                     )

    Parameters
    
    mod_preds - String, the input file paths for the validation, test set, hold out set predictions
    model_name - String, the name of the model to use when output the validation results
    
    Returns
    
    0 for successful execution
    
    Example
    
    model_validation(mod_preds = mod_preds, 
                     model_name = model_name
                     )

    """
    
    # extract out the validation output paths and the prediciton paths
    preds_valid_rmse = cons.result_output_paths['preds_valid_rmse'].format(model_type = model_name)
    preds_test_rmse = cons.result_output_paths['preds_test_rmse'].format(model_type = model_name)
    preds_vs_true_valid = cons.result_output_paths['preds_vs_true_valid'].format(model_type = model_name)
    preds_vs_true_test = cons.result_output_paths['preds_vs_true_test'].format(model_type = model_name)
    true_hist_valid = cons.result_output_paths['true_hist_valid'].format(model_type = model_name)
    true_hist_test = cons.result_output_paths['true_hist_test'].format(model_type = model_name)
    preds_hist_valid = cons.result_output_paths['preds_hist_valid'].format(model_type = model_name)
    preds_hist_test = cons.result_output_paths['preds_hist_test'].format(model_type = model_name)
    preds_hist_holdout = cons.result_output_paths['preds_hist_holdout'].format(model_type = model_name)
    y_valid_preds_path = cons.result_output_paths['y_valid_preds_path'].format(model_type = model_name)
    y_test_preds_path = cons.result_output_paths['y_test_preds_path'].format(model_type = model_name)
    y_holdout_preds_path = cons.result_output_paths['y_holdout_preds_path'].format(model_type = model_name)

    print('Loading model predictions ...')
    
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
    plot_preds_hist(dataset = y_valid, pred = 'item_cnt_day', bins = cons.bins, kde = cons.kde, model_name = model_name, out_fpath = true_hist_valid)
    plot_preds_hist(dataset = y_test, pred = 'item_cnt_day', bins = cons.bins, kde = cons.kde, model_name = model_name, out_fpath = true_hist_test)
    plot_preds_hist(dataset = y_valid, pred = 'y_valid_pred', bins = cons.bins, kde = cons.kde, model_name = model_name, out_fpath = preds_hist_valid)
    plot_preds_hist(dataset = y_test, pred = 'y_test_pred', bins = cons.bins, kde = cons.kde, model_name = model_name, out_fpath = preds_hist_test)
    plot_preds_hist(dataset = y_holdout, pred = 'y_holdout_pred', bins = cons.bins, kde = cons.kde, model_name = model_name, out_fpath = preds_hist_holdout)
    
    return 0
