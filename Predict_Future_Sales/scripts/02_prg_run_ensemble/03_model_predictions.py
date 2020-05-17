# -*- coding: utf-8 -*-
"""
Created on Sun May 17 16:00:32 2020

@author: oislen
"""

import pandas as pd
import utilities_ensemble as utl_ens
import pickle as pk

def model_preds(data_fpath,
                model_input_fpath,
                index_cols,
                tar_cols,
                pred_cols,
                data_splits_limits
                ):
    
    """
    """
    
    # load in model data
    base = pd.read_feather(model_input_fpath)

    # load best estimator here
    mod = pk.load(model_input_fpath)
    
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

    # make predictions for valid, test and holdout
    y_valid['y_valid_pred'] = mod.predict(X_valid[pred_cols])
    y_test['y_test_pred'] = mod.predict(X_test[pred_cols])
    y_holdout['y_holdout_pred'] = mod.predict(X_holdout[pred_cols])
    
    # output predictions
