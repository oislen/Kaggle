# -*- coding: utf-8 -*-
"""
Created on Sun May 17 16:00:32 2020

@author: oislen
"""

import pandas as pd
import utilities_ensemble as utl_ens
import joblib as jl

def model_preds(data_fpath,
                model_input_fpath,
                index_cols,
                tar_cols,
                pred_cols,
                data_splits_limits,
                pred_paths
                ):
    
    """
    """
    
    print('loading in base data ...')
    
    # load in model data
    base = pd.read_feather(data_fpath)

    print('loading in model ...')

    # load best estimator here
    mod = jl.load(model_input_fpath)
    
    # run the data splits function
    data_splits_dict = utl_ens.extract_data_splits(dataset = base,
                                                   index_cols = index_cols,
                                                   tar_cols = tar_cols,
                                                   pred_cols = pred_cols,
                                                   data_splits_limits = data_splits_limits
                                                   )
    
    # extract out the data splits
    #X_train = data_splits_dict['X_train']
    #y_train = data_splits_dict['y_train']
    X_valid = data_splits_dict['X_valid']
    y_valid = data_splits_dict['y_valid']
    X_test = data_splits_dict['X_test']
    y_test = data_splits_dict['y_test']
    X_holdout = data_splits_dict['X_holdout']
    y_holdout = data_splits_dict['y_holdout']

    print('making predictions ...')

    # make predictions for valid, test and holdout
    y_valid['y_valid_pred'] = mod.predict(X_valid[pred_cols])
    y_test['y_test_pred'] = mod.predict(X_test[pred_cols])
    y_holdout['y_holdout_pred'] = mod.predict(X_holdout[pred_cols])
    
    print('outputting predctions ..')
    
    # extract out the prediciton paths
    y_valid_preds_path = pred_paths['y_valid_preds_path']
    y_test_preds_path = pred_paths['y_test_preds_path']
    y_holdout_preds_path = pred_paths['y_holdout_preds_path']
    
    # output predictions
    y_valid.to_csv(y_valid_preds_path, index = False)
    y_test.to_csv(y_test_preds_path, index = False)
    y_holdout.to_csv(y_holdout_preds_path, index = False)

    return