# -*- coding: utf-8 -*-
"""
Created on Sun May 17 12:20:21 2020

@author: oislen
"""

from importlib import import_module
from reference.extract_feat_imp import extract_feat_imp

model_train = import_module(name = '01_model_training')
model_pred = import_module(name = '02_model_predictions')
model_valid = import_module(name = '03_model_validation')
kaggl_pred = import_module(name = '04_format_kaggle_preds')

def exe_model(cons,
              feat_imp,
              n,
              skip_train,
              n_cpu,
              model_type,
              max_dept,
              date,
              rand_state
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
    
    exe_model(cons,
              feat_imp,
              n,
              skip_train,
              n_cpu,
              model_type,
              max_dept,
              date,
              rand_state
              )
    
    Parameters
    
    cons - Python Module, the compeition programme constants
    feat_imp - String, the type of model feature importance to use
    n - Integer, the number of top ranked features to extract from the feature importance results
    skip_train - Boolean, whether to skip the model training set, if models are already trained from previous iteration
    n_cpu - Integer, the number of cpus / jobs to use when performing the grid search cross validation to find optimal model parameters
    model_type - String, the type of model being used to generate model predictions
    max_dept - Integer, the maximum dept for tree based models
    date - String, the date to use when outputing model results
    rand_state - Integer, the random seed to use for reproducabilty
    
    Returns
    
    0 for successful execution
    
    Example
    
    exe_model(cons = cons, 
              feat_imp = 'randforest', 
              max_dept = 3, 
              rand_state = 1, 
              model_type = 'dtree', 
              n = 30, 
              skip_train = False, 
              n_cpu = -1, 
              date = '20200502'
              )
        
    
    """
    
    model_name = cons.model_name.format(model_type = model_type, max_dept = max_dept)
    mod_preds = cons.mod_preds.format(pred_data_dir = cons.pred_data_dir, model_name = model_name, date = date)
    model_pk_fpath = cons.model_pk_fpath.format(models_dir = cons.models_dir, model_name = model_name)
    cv_sum_fpath = cons.cv_sum_fpath.format(cv_results_dir = cons.cv_results_dir, model_name = model_name)
    model = cons.model_dict[model_type]
    model_params = cons.params_dict[model_type]
    data_fpath = cons.model_data_fpath
    train_cv_split_dict = cons.train_cv_split_dict
    test_split_dict = cons.test_split_dict
    
    # assign additional model parameters
    model_params['max_depth'] = [max_dept]
    model_params['random_state'] = [rand_state]
    if model_type == 'randforest':
        model_params['n_jobs'] = [n_cpu]
        
    
    # TODO functionise this 
    # set the prediction output paths
    y_valid_preds_path = '{}_valid.feather'.format(mod_preds)
    y_test_preds_path = '{}_test.feather'.format(mod_preds)
    y_holdout_preds_path = '{}_holdout.feather'.format(mod_preds)
    meta_lvl_II_feats_path = '{}_meta_lvl_II_feats.feather'.format(mod_preds)
    kaggle_preds = '{}.csv'.format(mod_preds)
    
    # set final predictions
    pred_paths = {'y_valid_preds_path':y_valid_preds_path,
                  'y_test_preds_path':y_test_preds_path,
                  'y_holdout_preds_path':y_holdout_preds_path,
                  'meta_lvl_II_feats_path':meta_lvl_II_feats_path,
                  'kaggle_preds':kaggle_preds
                  }
    
    
    # TODO: functionise this
    # set validation file path
    preds_vs_true_fpath = '{}/{}'.format(cons.valid_preds_vs_true_dir, model_name)
    preds_hist_fpath = '{}/{}'.format(cons.valid_preds_hist_dir, model_name)
    preds_metrics_fpath = '{}/{}'.format(cons.valid_metrics_dir, model_name)

    # set the validation output paths
    preds_valid_rmse = '{}_valid_rmse.csv'.format(preds_metrics_fpath)
    preds_test_rmse = '{}_test_rmse.csv'.format(preds_metrics_fpath)
    preds_vs_true_valid = '{}_preds_vs_true_valid.png'.format(preds_vs_true_fpath)
    preds_vs_true_test = '{}_preds_vs_true_test.png'.format(preds_vs_true_fpath)
    true_hist_valid = '{}_true_valid.png'.format(preds_hist_fpath)
    true_hist_test = '{}_true_test.png'.format(preds_hist_fpath)
    preds_hist_valid = '{}_preds_valid.png'.format(preds_hist_fpath)
    preds_hist_test = '{}_preds_test.png'.format(preds_hist_fpath)
    preds_hist_holdout = '{}_preds_holdout.png'.format(preds_hist_fpath)
    
    # create a dictionary for the validation output file paths
    valid_output_paths = {'preds_valid_rmse':preds_valid_rmse,
                          'preds_test_rmse':preds_test_rmse,
                          'preds_vs_true_valid':preds_vs_true_valid,
                          'preds_vs_true_test':preds_vs_true_test,
                          'true_hist_valid':true_hist_valid,
                          'true_hist_test':true_hist_test,
                          'preds_hist_valid':preds_hist_valid,
                          'preds_hist_test':preds_hist_test,
                          'preds_hist_holdout':preds_hist_holdout
                          }
    
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
                                   cv_sum_fpath = cv_sum_fpath,
                                   n_cpu = n_cpu
                                   )
        
    # get model predictions
    model_pred.model_preds(data_fpath = data_fpath,
                           model_pk_fpath = model_pk_fpath,
                           index_cols = index_cols,
                           req_cols = req_cols,
                           tar_cols = tar_cols,
                           pred_cols = pred_cols,
                           test_split_dict = test_split_dict,
                           pred_paths = pred_paths
                           )
    
    # call model validation
    model_valid.model_validation(pred_paths = pred_paths,
                                 valid_output_paths = valid_output_paths,
                                 model_name = model_name
                                 )
    
    # call the kaggle format predictions
    kaggl_pred.format_kaggle_preds(pred_paths = pred_paths)

    return 0