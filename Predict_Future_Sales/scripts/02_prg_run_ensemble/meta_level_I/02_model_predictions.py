# -*- coding: utf-8 -*-
"""
Created on Sun May 17 16:00:32 2020

@author: oislen
"""

import pandas as pd
import utilities_ensemble as utl_ens
import joblib as jl

def model_preds(data_fpath,
                model_pk_fpath,
                index_cols,
                tar_cols,
                pred_cols,
                test_split_dict,
                pred_paths
                ):
    
    """
    """
    
    print('loading in base data ...'.format(data_fpath))
    
    # load in model data
    base = pd.read_feather(data_fpath)

    print('loading in model {} ...'.format(model_pk_fpath))

    # load best estimator here
    mod = jl.load(model_pk_fpath)
    
    # run the data splits function
    data_splits_dict = utl_ens.extract_data_splits(dataset = base,
                                                   index_cols = index_cols,
                                                   tar_cols = tar_cols,
                                                   pred_cols = pred_cols,
                                                   test_split_dict = test_split_dict
                                                   )
    
    # extract out the data splits
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
    
    #print('format model predictions ...')
    
    #y_valid = utl_ens.format_preds(dataset = y_valid, preds_cols = 'y_valid_pred')
    #y_test = utl_ens.format_preds(dataset = y_test, preds_cols = 'y_test_pred')
    #y_holdout = utl_ens.format_preds(dataset = y_holdout, preds_cols = 'y_holdout_pred')
    
    print('outputting predctions ..')
    
    # extract out the prediciton paths
    y_valid_preds_path = pred_paths['y_valid_preds_path']
    y_test_preds_path = pred_paths['y_test_preds_path']
    y_holdout_preds_path = pred_paths['y_holdout_preds_path']
    
    # output predictions
    y_valid.to_feather(y_valid_preds_path, index = False)
    y_test.to_feather(y_test_preds_path, index = False)
    y_holdout.to_feather(y_holdout_preds_path, index = False)

    return