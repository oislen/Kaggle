# -*- coding: utf-8 -*-
"""
Created on Sun May 17 12:27:58 2020

@author: oislen
"""

import os
import cons
import pandas as pd
from calc_rmse import calc_rmse
from plot_preds_vs_true import plot_preds_vs_true
from hist import hist

def model_validation(pred_paths, 
                     valid_output_paths,
                     model_name
                     ):
    
    """
    
    Model Validation Documentation
    
    Function Overview
    
    This function performs validation checks on the predictions for the validation, test set and hold out set.
    
    Defaults
    
    model_validation(pred_paths, 
                     valid_output_paths,
                     model_name
                     )

    Parameters
    
    pred_paths - Dictionary, the input file paths for the validation, test set, hold out set predictions
    valid_output_paths - Dictionary, the output file paths for the validation, test set and holdout set validation results
    model_name - String, the name of the model to use when output the validation results
    
    Returns
    
    0 for successful execution
    
    Example
    
    model_validation(pred_paths = pred_paths, 
                     valid_output_paths = valid_output_paths,
                     model_name = model_name
                     )

    """
    
    # extract out the validation output paths
    preds_valid_rmse = valid_output_paths['preds_valid_rmse'].format(model_type = model_name)
    preds_test_rmse = valid_output_paths['preds_test_rmse'].format(model_type = model_name)
    preds_vs_true_valid = valid_output_paths['preds_vs_true_valid'].format(model_type = model_name)
    preds_vs_true_test = valid_output_paths['preds_vs_true_test'].format(model_type = model_name)
    true_hist_valid = valid_output_paths['true_hist_valid'].format(model_type = model_name)
    true_hist_test = valid_output_paths['true_hist_test'].format(model_type = model_name)
    preds_hist_valid = valid_output_paths['preds_hist_valid'].format(model_type = model_name)
    preds_hist_test = valid_output_paths['preds_hist_test'].format(model_type = model_name)
    preds_hist_holdout = valid_output_paths['preds_hist_holdout'].format(model_type = model_name)

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
    hist(dataset = y_valid, num_var = ['item_cnt_day'], bins = cons.bins, kde = cons.kde, title = model_name, output_dir = os.path.dirname(true_hist_valid), output_fname = os.path.basename(true_hist_valid))
    hist(dataset = y_test, num_var = ['item_cnt_day'], bins = cons.bins, kde = cons.kde, title = model_name, output_dir = os.path.dirname(true_hist_test), output_fname = os.path.basename(true_hist_test))
    hist(dataset = y_valid, num_var = ['y_valid_pred'], bins = cons.bins, kde = cons.kde, title = model_name, output_dir = os.path.dirname(preds_hist_valid), output_fname = os.path.basename(preds_hist_valid))
    hist(dataset = y_test, num_var = ['y_test_pred'], bins = cons.bins, kde = cons.kde, title = model_name, output_dir = os.path.dirname(preds_hist_test), output_fname = os.path.basename(preds_hist_test))
    hist(dataset = y_holdout, num_var = ['y_holdout_pred'], bins = cons.bins, kde = cons.kde, title = model_name, output_dir = os.path.dirname(preds_hist_holdout), output_fname = os.path.basename(preds_hist_holdout))
    
    return 0
