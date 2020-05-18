# -*- coding: utf-8 -*-
"""
Created on Sun May 17 12:20:21 2020

@author: oislen
"""

from sklearn.ensemble import RandomForestRegressor
from importlib import import_module
import utilities_ensemble as utl_ens

model_train = import_module(name = '02_model_training')
model_pred = import_module(name = '03_model_predictions')
model_valid = import_module(name = '04_model_validation')
kaggl_pred = import_module(name = '05_format_kaggle_preds')

def mod_randfrest(cons):
    
    """
    """
    
    #~~~~~~~~ User Inputs ~~~~~~~~~~~~~~~~~~~~~~~~~
    
    model_type = 'randforest'
    
    model_params = {'criterion':['mse'],
                    'max_depth':[7],
                    'random_state':[1234],
                    'n_estimators':[100],
                    'max_features':['auto']
                    }

    
    # initiate random forest model
    rfc = RandomForestRegressor()

    # set the train, valid and test sub limits
    cv_split_dict = [{'train_sub':30, 'valid_sub':31}]
    
    # set model pk output file path
    model_pk_output_fpath = '{}/randforest_mode.pickle'.format(cons.models_dir)
    
    # set the train, valid and test sub limits
    data_splits_limits = {'train_sub':31, 'valid_sub':32, 'test_sub':33}
    
    # set the output paths
    y_valid_preds_path = cons.randforest_preds + '_valid.csv'
    y_test_preds_path = cons.randforest_preds + '_test.csv'
    y_holdout_preds_path = cons.randforest_preds + '_holdout.csv'
    
    pred_paths = {'y_valid_preds_path':y_valid_preds_path,
                  'y_test_preds_path':y_test_preds_path,
                  'y_holdout_preds_path':y_holdout_preds_path
                  }
    
    
    # set final predictions
    kaggle_preds = cons.randforest_preds + '.csv'
    
    # set the input data file path
    data_fpath = cons.model_data_fpath
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # load in feature importance cols
    extract_feat_imp = utl_ens.extract_feat_imp(cons = cons, 
                                                model_type = model_type,
                                                n = 30
                                                )
    
    # set the target abd predictors to tune
    index_cols = extract_feat_imp['index_cols']
    tar_cols = extract_feat_imp['tar_cols']
    pred_cols = extract_feat_imp['pred_cols']
 
    # run cv model training
    model_train.cv_model_train(data_fpath = data_fpath,
                               tar_cols = tar_cols,
                               pred_cols = pred_cols,
                               model = rfc,
                               model_params = model_params,
                               cv_split_dict = cv_split_dict,
                               model_output_fpath = model_pk_output_fpath
                               )
    
    
    # get model predictions
    model_pred.model_preds(data_fpath = data_fpath,
                           model_input_fpath = model_pk_output_fpath,
                           index_cols = index_cols,
                           tar_cols = tar_cols,
                           pred_cols = pred_cols,
                           data_splits_limits = data_splits_limits,
                           pred_paths = pred_paths
                           )
    
    # call model validation
    model_valid.model_validation(pred_paths = pred_paths)
    
    # call the kaggle format predictions
    kaggl_pred.format_kaggle_preds(pred_paths = pred_paths,
                                   kaggle_preds = kaggle_preds
                                   )

    return