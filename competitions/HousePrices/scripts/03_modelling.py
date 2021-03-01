# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 09:58:04 2019

@author: oislen
"""

# load in the relevant libraries
import cons
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from model.perf_metrics import perf_metrics
from model.StackModels import StackModels

def model_data(data_fpath):
    
    """
    
    Model Data Documentation
    
    Function Overview
    
    Defaults
    
    Parameters
    
    Returns
    
    Example
    
    """
    
    print('Loading in data ...')
    
    # load in data
    clean = pd.read_csv(data_fpath, 
                        sep = cons.sep
                        )
    
    print('Splitting data ...')
    
    # extract out the test set id column
    test_ID = clean.loc[(clean['Dataset'] == 'test'), 'Id']
    
    # split the datasets
    test = clean[(clean['Dataset'] == 'test')]
    train = clean[(clean['Dataset'] == 'train')]
    
    # extract out the columns
    y_col = 'logSalePrice'
    X_cols = train.columns.drop(['Id', 'logSalePrice', 'Dataset'])
    
    # randomly split the dataset
    X_valid, X_train, y_valid, y_train = train_test_split(train[X_cols],
                                                          train[y_col],
                                                          train_size = 0.8,
                                                          test_size = 0.2
                                                          )
    
    # extract out the relevant columns
    y_train_sub = y_train.values
    X_test = test.drop(columns = ['Id', 'logSalePrice', 'Dataset'])
    
    # for each model
    for name, model in cons.models_dict.items():
        
        print('Working on ' + name)
        
        # generate cross fold validation metrics
        perf_metrics(model = model, 
                     endog = y_train_sub,
                     exog = X_train,
                     target_type = 'reg',
                     n_folds = 5,
                     digits = 3,
                     output_dir = cons.metrics_dir,
                     output_fname = '{}_model_metrics.csv'.format(name)
                     )
    
    #-- Stack Model --#
    
    print('Creating stacked model ...')
    
    # create the model (via stacked class)
    StackMod = StackModels(base_models = (cons.ENet, cons.GBoost, cons.KRR),
                           meta_model = cons.lasso
                           )
    
    # fit the model
    StackMod.fit(X_train.values, y_train_sub)
    
    # predict for the validation values
    sm_y_valid_pred = StackMod.predict(X_valid.values)
    
    # generate metrics for the predictions
    perf_metrics(y_obs = y_valid.values,
                 y_pred = sm_y_valid_pred,
                 target_type = 'reg',
                 digits = 4
                 )
    
    #-- XGBoost Model --#
    
    print('Creating XGBoost model ...')
    
    # extract out xgb model
    model_xgb = cons.model_xgb
    
    # fit the xgboost mode
    model_xgb.fit(X_train.values, y_train_sub)
    
    # predict for the validation values
    xgb_y_valid_pred = model_xgb.predict(X_valid.values)
    
    # generate metrics for the predictions
    perf_metrics(y_obs = y_valid.values,
                 y_pred = xgb_y_valid_pred,
                 target_type = 'reg',
                 digits = 4
                 )
    
    #-- LG Boost Model --#
    
    print('Creating LGBoost Model ...')
    
    # extract out lgb model
    model_lgb = cons.model_lgb
    
    # fit the lgboost model
    model_lgb.fit(X_train.values, y_train_sub)
    
    # predict of the validation values
    lgb_y_valid_pred = model_lgb.predict(X_valid.values)
    
    # generate metrics for the predictions
    perf_metrics(y_obs = y_valid.values,
                 y_pred = lgb_y_valid_pred,
                 target_type = 'reg',
                 digits = 4
                 )
    
    #-- Weighted Model --#
    
    print('Generate weighted predictions ...')
    
    #  set up stacked model
    drop_cols = ['Id', 'Dataset', 'logSalePrice']
    X_train_full = train.drop(columns = drop_cols).values
    y_train_full = train['logSalePrice'].values
    X_test = test.drop(columns = drop_cols).values
    
    # create the stacked model predictions
    StackMod.fit(X_train_full, y_train_full)
    sm_pred = np.exp(StackMod.predict(X_test))
    
    # create the xgboost model predictions
    model_xgb.fit(X_train_full, y_train_full)
    xgb_pred = np.exp(model_xgb.predict(X_test))
    
    # create the lgboost model predictions
    model_lgb.fit(X_train_full, y_train_full)
    lgb_pred = np.exp(model_lgb.predict(X_test))
    
    # create a weighted ensemble from the three models
    ensemble = sm_pred * 0.70 + xgb_pred * 0.15 + lgb_pred * 0.15
    
    print('Creating output predictions ...')
    
    # create submission file
    sub = pd.DataFrame()
    sub['Id'] = test_ID.astype(int)
    sub['SalePrice'] = ensemble
    
    # output the dataset
    sub.to_csv(cons.preds_data_fpath,
               sep = cons.sep,
               encoding = cons.encoding,
               header = cons.header,
               index = cons.index
               )
    
    return 0

# if running script as main programme
if __name__ == '__main__':
    
    # extract out the file path
    data_fpath = cons.engin_data_fpath
    
    # execute model
    model_data(data_fpath = data_fpath)