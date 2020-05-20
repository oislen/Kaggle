# -*- coding: utf-8 -*-
"""
Created on Mon May 18 20:42:41 2020

@author: oislen
"""

from sklearn.ensemble import RandomForestRegressor
from exe_model import exe_model

def mod_randforest(cons):
    
    # set model name
    model_type = 'randforest'
    
    # set model parameters
    model_params = {'criterion':['mse'],
                    'max_depth':[7],
                    'random_state':[1234],
                    'n_estimators':[10],
                    'max_features':['auto']
                    }
    
    
    # initiate random forest model
    rfc = RandomForestRegressor()
    
    # set the train, valid and test sub limits
    train_cv_split_dict = [{'train_sub':30, 'valid_sub':31}]
    
    # set model pk output file path
    model_pk_fpath = '{}/randforest_mode.pickle'.format(cons.models_dir)
    
    # set the train, valid and test sub limits
    test_split_dict = {'train_sub':31, 'valid_sub':32, 'test_sub':33}
    
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
    
    # set the number of input features
    n = 35
    
    # execute the model
    exe_model(cons = cons,
              model_type = model_type,
              data_fpath = data_fpath,
              model = rfc,
              model_params = model_params,
              train_cv_split_dict = train_cv_split_dict,
              model_pk_fpath = model_pk_fpath,
              test_split_dict = test_split_dict,
              pred_paths = pred_paths,
              kaggle_preds = kaggle_preds,
              n = n
              )