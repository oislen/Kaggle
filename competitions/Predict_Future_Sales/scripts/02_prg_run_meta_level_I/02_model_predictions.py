# -*- coding: utf-8 -*-
"""
Created on Sun May 17 16:00:32 2020

@author: oislen
"""

import cons
import pandas as pd
import joblib as jl
from reference.extract_data_splits import extract_data_splits

def model_preds(data_fpath,
                model_pk_fpath,
                index_cols,
                req_cols,
                tar_cols,
                pred_cols,
                test_split_dict,
                model_name
                ):
    
    """
    
    Model Predictions Documentation
    
    Function Overview
    
    This function generates predictions using the fitted models for the validation, test set, holdout set and meta level II set.
    
    Defaults
    
    model_preds(data_fpath,
                model_pk_fpath,
                index_cols,
                req_cols,
                tar_cols,
                pred_cols,
                test_split_dict,
                pred_paths,
                model_name
                )
    
    Parameters
    
    data_fpath - String, the full file path to the base data
    model_pk_fpath - String, the full file path to the fitted model
    index_cols - List of Strings, the index columns of the base data
    req_cols - List of Strings, the required columns of the base data
    tar_cols - List of Strings, the target columns of the base data
    pred_cosl - List of Strings, the predictor columns of the base data
    test_split_dict - Dictionary, the train, validation and test splitting configurations for the modelling data
    model_name - String, the name of the model
    
    Returns
    
    0 for successful execution
    
    Example
    
    model_preds(data_fpath = data_fpath,
                model_pk_fpath = model_pk_fpath,
                index_cols = index_cols,
                req_cols = req_cols,
                tar_cols = tar_cols,
                pred_cols = pred_cols,
                test_split_dict = {'train_sub':29, 'valid_sub':32, 'test_sub':33}
                )
    
    """
    
    print('loading in base data {} ...'.format(data_fpath))
    
    # load in model data
    base = pd.read_feather(data_fpath)

    print('loading in model {} ...'.format(model_pk_fpath))

    # load best estimator here
    mod = jl.load(model_pk_fpath)
    
    print('splitting out dataset ...')
    
    # run the data splits function
    data_splits_dict = extract_data_splits(dataset = base,
                                           index_cols = index_cols,
                                           tar_cols = tar_cols,
                                           req_cols = req_cols,
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
    X_meta_lvl_II = data_splits_dict['X_meta_lvl_II']
    y_meta_lvl_II = data_splits_dict['y_meta_lvl_II']
    
    print('making predictions ...')

    print(X_valid[pred_cols].head())
    print(y_test.head())

    print(pred_cols)
    
    # make predictions for valid, test, holdout and meta lvl II
    y_valid['y_valid_pred'] = mod.predict(X_valid[pred_cols])
    y_test['y_test_pred'] = mod.predict(X_test[pred_cols])
    y_holdout['y_holdout_pred'] = mod.predict(X_holdout[pred_cols])
    y_meta_lvl_II['y_meta_lvl_I_pred'] = mod.predict(X_meta_lvl_II[pred_cols])
    
    print('outputting predctions ..')
    
    # extract out the prediciton paths
    y_valid_preds_path = cons.result_output_paths['y_valid_preds_path'].format(model_type = model_name)
    y_test_preds_path = cons.result_output_paths['y_test_preds_path'].format(model_type = model_name)
    y_holdout_preds_path = cons.result_output_paths['y_holdout_preds_path'].format(model_type = model_name)
    meta_lvl_II_feats_path = cons.result_output_paths['meta_lvl_II_feats_path'].format(model_type = model_name)
    
    # output predictions
    y_valid.reset_index(drop = True).to_feather(y_valid_preds_path)
    y_test.reset_index(drop = True).to_feather(y_test_preds_path)
    y_holdout.reset_index(drop = True).to_feather(y_holdout_preds_path)
    y_meta_lvl_II.reset_index(drop = True).to_feather(meta_lvl_II_feats_path)

    return 0