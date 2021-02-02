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
              mod_preds,
              cv_sum_fpath,
              test_split_dict,
              n,
              model_name,
              skip_train,
              n_cpu
              ):
    
    """
    
    Execute Level I Model Documentation
    
    """
    
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
    extract_feat_imp = utl_ens.extract_feat_imp(cons = cons, 
                                                feat_imp = feat_imp,
                                                n = n
                                                )
    
    # set the target abd predictors to tune
    index_cols = extract_feat_imp['index_cols']
    req_cols = extract_feat_imp['req_cols']
    tar_cols = extract_feat_imp['tar_cols']
    pred_cols = extract_feat_imp['pred_cols']
    
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

    return