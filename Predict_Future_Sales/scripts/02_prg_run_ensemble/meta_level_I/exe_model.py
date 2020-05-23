# -*- coding: utf-8 -*-
"""
Created on Sun May 17 12:20:21 2020

@author: oislen
"""

from importlib import import_module
import utilities_ensemble as utl_ens

model_train = import_module(name = '01_model_training')
model_pred = import_module(name = '02_model_predictions')
model_valid = import_module(name = '03_model_validation')
kaggl_pred = import_module(name = '04_format_kaggle_preds')

def exe_model(cons,
              feat_imp,
              data_fpath,
              model,
              model_params,
              train_cv_split_dict,
              model_pk_fpath,
              cv_sum_fpath,
              test_split_dict,
              pred_paths,
              n = 35
              ):
    
    """
    
    Execute Level I Model Documentation
    
    """

    # load in feature importance cols
    extract_feat_imp = utl_ens.extract_feat_imp(cons = cons, 
                                                feat_imp = feat_imp,
                                                n = n
                                                )
    
    # set the target abd predictors to tune
    index_cols = extract_feat_imp['index_cols']
    tar_cols = extract_feat_imp['tar_cols']
    pred_cols = extract_feat_imp['pred_cols']
 
    # run cv model training
    model_train.cv_model_train(data_fpath = data_fpath,
                               tar_cols = tar_cols,
                               pred_cols = pred_cols,
                               model = model,
                               model_params = model_params,
                               train_cv_split_dict = train_cv_split_dict,
                               model_pk_fpath = model_pk_fpath,
                               cv_sum_fpath = cv_sum_fpath
                               )
    
    # get model predictions
    model_pred.model_preds(data_fpath = data_fpath,
                           model_pk_fpath = model_pk_fpath,
                           index_cols = index_cols,
                           tar_cols = tar_cols,
                           pred_cols = pred_cols,
                           test_split_dict = test_split_dict,
                           pred_paths = pred_paths
                           )
    
    # call model validation
    model_valid.model_validation(pred_paths = pred_paths)
    
    # call the kaggle format predictions
    kaggl_pred.format_kaggle_preds(pred_paths = pred_paths)

    return