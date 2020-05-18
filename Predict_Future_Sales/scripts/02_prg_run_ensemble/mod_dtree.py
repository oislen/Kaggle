# -*- coding: utf-8 -*-
"""
Created on Sun May 17 12:20:21 2020

@author: oislen
"""

from sklearn.tree import DecisionTreeRegressor
from exe_model import exe_model
import numpy as np

def mod_dtree(cons):
    
    """
    """
    
    # set model name
    model_type = 'randforest'
    
    # initiate decision tree regressor
    dtr = DecisionTreeRegressor()
    
    # set model parameters
    model_params = {'criterion':['mse'],
                    'splitter':['best'],
                    'max_depth':[3, 5, 7, 9],
                    'max_features':['auto'],
                    'random_state':[1234]
                    }
    
    
    # set the train, valid and test sub limits
    train_cv_split_dict = [{'train_sub':idx, 'valid_sub':idx + 1} for idx in np.arange(start = 1, stop = 30, step = 12)]
        
    # set model pk output file path
    model_pk_fpath = '{}/dtree_mode.pickle'.format(cons.models_dir)
    
    # set the train, valid and test sub limits
    test_split_dict = {'train_sub':31, 'valid_sub':32, 'test_sub':33}
    
    # set the output paths
    y_valid_preds_path = cons.dtree_preds + '_valid.csv'
    y_test_preds_path = cons.dtree_preds + '_test.csv'
    y_holdout_preds_path = cons.dtree_preds + '_holdout.csv'
    
    pred_paths = {'y_valid_preds_path':y_valid_preds_path,
                  'y_test_preds_path':y_test_preds_path,
                  'y_holdout_preds_path':y_holdout_preds_path
                  }
    
    
    # set final predictions
    kaggle_preds = cons.dtree_preds + '.csv'
    
    # set the input data file path
    data_fpath = cons.model_data_fpath
    
    # execute the model
    exe_model(cons = cons,
              model_type = model_type,
              data_fpath = data_fpath,
              model = dtr,
              model_params = model_params,
              train_cv_split_dict = train_cv_split_dict,
              model_pk_fpath = model_pk_fpath,
              test_split_dict = test_split_dict,
              pred_paths = pred_paths,
              kaggle_preds = kaggle_preds
              )

    return