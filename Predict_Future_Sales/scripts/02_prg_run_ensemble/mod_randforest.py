# -*- coding: utf-8 -*-
"""
Created on Sun May 17 12:20:21 2020

@author: oislen
"""

from sklearn.ensemble import RandomForestRegressor
from importlib import import_module

model_train = import_module(name = '02_model_training')
model_pred = import_module(name = '03_model_predictions')

def mod_randfrest(cons):
    
    """
    """
    
    # initiate random forest model
    rfc = RandomForestRegressor()
 
    # set the target abd predictors to tune
    index_cols = []
    tar_cols = []
    pred_cols = []
 
    model_params = {'criterion':['mse'],
                    'max_depth':[7],
                    'random_state':[1234],
                    'n_estimators':[100],
                    'max_features':['auto']
                    }
    
    # set the input data file path
    data_fpath = cons.model_data_fpath
    
    # set the train, valid and test sub limits
    cv_split_dict = {'train_sub':32, 'valid_sub':33} 
    
    # set model pk output file path
    model_pk_output_fpath = '{}/randforest_mode.pickle'.format(cons.models_dir)
    
    # run cv model training
    model_train.cv_model_train(data_fpath = data_fpath,
                               tar_cols = tar_cols,
                               pred_cols = pred_cols,
                               model = rfc,
                               model_params = model_params,
                               cv_split_dict = cv_split_dict,
                               model_output_fpath = model_pk_output_fpath
                               )
    
    
    # set the train, valid and test sub limits
    data_splits_limits = {'train_sub':31,
                          'valid_sub':32,
                          'test_sub':33
                          }

    
    # get model predictions
    model_pred.model_preds(data_fpath = data_fpath,
                           model_input_fpath = model_pk_output_fpath,
                           index_cols = index_cols,
                           tar_cols = tar_cols,
                           pred_cols = pred_cols,
                           data_splits_limits = data_splits_limits
                           )

