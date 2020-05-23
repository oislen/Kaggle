# -*- coding: utf-8 -*-
"""
Created on Wed May 20 09:30:46 2020

@author: oislen
"""



from sklearn.ensemble import GradientBoostingRegressor
from meta_level_I import exe_model

def mod_gradboost(cons):
    
    # set model name
    model_type = 'gradboost'
    
    # set model parameters
    model_params = {'criterion':['mse'],
                    'max_depth':[3],
                    'random_state':[1234],
                    'n_estimators':[10],
                    'max_features':['auto']
                    }
    
    
    # initiate random forest model
    gbr = GradientBoostingRegressor()
    
    # set the train, valid and test sub limits
    train_cv_split_dict = [{'train_sub':30, 'valid_sub':31}]
    
    # set model pk output file path
    model_pk_fpath = '{}/gradboost_mode.pickle'.format(cons.models_dir)
    
    # set the train, valid and test sub limits
    test_split_dict = {'train_sub':31, 'valid_sub':32, 'test_sub':33}
    
    # set the output paths
    y_valid_preds_path = cons.gradboost_preds + '_valid.csv'
    y_test_preds_path = cons.gradboost_preds + '_test.csv'
    y_holdout_preds_path = cons.gradboost_preds + '_holdout.csv'
    
    pred_paths = {'y_valid_preds_path':y_valid_preds_path,
                  'y_test_preds_path':y_test_preds_path,
                  'y_holdout_preds_path':y_holdout_preds_path
                  }
    
    # set final predictions
    kaggle_preds = cons.gradboost_preds + '.csv'
    
    # set the input data file path
    data_fpath = cons.model_data_fpath
    
    #set the number of features to use
    n = 35
    
    # execute the model
    exe_model(cons = cons,
              model_type = model_type,
              data_fpath = data_fpath,
              model = gbr,
              model_params = model_params,
              train_cv_split_dict = train_cv_split_dict,
              model_pk_fpath = model_pk_fpath,
              test_split_dict = test_split_dict,
              pred_paths = pred_paths,
              kaggle_preds = kaggle_preds,
              n = n
              )