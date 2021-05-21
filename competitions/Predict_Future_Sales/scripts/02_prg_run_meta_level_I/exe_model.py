# -*- coding: utf-8 -*-
"""
Created on Sun May 17 12:20:21 2020

@author: oislen
"""

import cons
from importlib import import_module
from reference.extract_feat_imp import extract_feat_imp

model_train = import_module(name = '01_model_training')
model_pred = import_module(name = '02_model_predictions')
model_valid = import_module(name = '03_model_validation')
kaggl_pred = import_module(name = '04_format_kaggle_preds')

def exe_model(feat_imp,
              n,
              skip_train,
              model_type
              ):
    
    """
    
    Execute Level I Model Documentation
    
    Function Overview
    
    This function executes the meta lebel I model steps:
        1. Training the initial sklearn model using GridSearchcv.
        2. Making model predictions for validation set, out-of-sample test set and kaggle competition test file.
        3. Validating model predictions for validation set and out-of-sample test set.
        4. Formating the kaggle competition test file
    
    Defaults
    
    exe_model(feat_imp,
              n,
              skip_train,
              model_type,
              max_dept
              )
    
    Parameters
    
    feat_imp - String, the type of model feature importance to use
    n - Integer, the number of top ranked features to extract from the feature importance results
    skip_train - Boolean, whether to skip the model training set, if models are already trained from previous iteration
    model_type - String, the type of model being used to generate model predictions
    
    Returns
    
    0 for successful execution
    
    Example
    
    exe_model(feat_imp = 'randforest', 
              model_type = 'dtree', 
              n = 30, 
              skip_train = False
              )
        
    
    """
    
    model_name = cons.model_name.format(model_type = model_type)
    model_pk_fpath = cons.model_pk_fpath.format(models_dir = cons.models_dir, model_name = model_name)
    cv_sum_fpath = cons.cv_sum_fpath.format(cv_results_dir = cons.cv_results_dir, model_name = model_name)
    model = cons.model_dict[model_type]
    model_params = cons.params_dict[model_type]
    data_fpath = cons.model_data_fpath
    train_cv_split_dict = cons.train_cv_split_dict
    test_split_dict = cons.test_split_dict
    
    # assign additional model parameters
    if model_type == 'randforest':
        model_params['n_jobs'] = [cons.n_cpu]
    
    # load in feature importance cols
    extract_feat_imp_df = extract_feat_imp(cons = cons, 
                                           feat_imp = feat_imp,
                                           n = n
                                           )
    
    # set the target abd predictors to tune
    index_cols = extract_feat_imp_df['index_cols']
    req_cols = extract_feat_imp_df['req_cols']
    tar_cols = extract_feat_imp_df['tar_cols']
    pred_cols = extract_feat_imp_df['pred_cols']
    
    # if skipping the training step
    if not skip_train:
        
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
                           req_cols = req_cols,
                           tar_cols = tar_cols,
                           pred_cols = pred_cols,
                           test_split_dict = test_split_dict,
                           model_name = model_name
                           )
    
    # call model validation
    model_valid.model_validation(model_name = model_name)
    
    # call the kaggle format predictions
    kaggl_pred.format_kaggle_preds(model_name = model_name)

    return 0