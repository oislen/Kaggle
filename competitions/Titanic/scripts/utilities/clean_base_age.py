# -*- coding: utf-8 -*-
"""
Created on Sun Nov  4 19:18:53 2018

@author: oislen
"""

# load in relevant libraries
import pandas as pd
import cons
from sklearn import ensemble
import value_analysis as va

def clean_base_age(base_clean_fpath,
                   base_clean_2_fpath
                   ):
        
    """

    Clean Base Age Documentation

    Function Overview
    
    This function cleans and processes the missing age values.
    
    Defaults
    
    clean_base_age(base_clean_fpath,
                   base_clean_2_fpath
                   )
    
    Parameters
    
    base_clean_fpath - String, the input file path to the cleaned base data
    base_clean_2_fpath - String, the output file path to write the cleaned aged base data
    
    Returns
    
    0 fro successful execution
    
    Example

    clean_base_age(base_clean_fpath = 'C:\\Users\\...\\base_clean.csv',
                   base_clean_2_fpath = 'C:\\Users\\...\\base_clean_2.csv'
                   )

    """
    
    # load in data
    base = pd.read_csv(base_clean_fpath, sep = '|')
    
    # check data types
    base.dtypes
    
    # split the data based on the original dataset
    base_train = base[base.Dataset == 'train']
    base_test = base[base.Dataset == 'test']

    print('Cleaning Age in base trian ...')
    
    # split the training data on whether age is missing or not
    train = base_train[base_train.Age.notnull()]
    test = base_train[base_train.Age.isnull()]
    
    # randomly split the dataset
    X_valid, X_train, y_valid, y_train = va.train_test_split_sample(dataset = train,
                                                                    y = ['Age'],
                                                                    X = train.columns.drop(['Age', 'PassengerId', 'Dataset']),
                                                                    train_size = 0.8,
                                                                    test_size = 0.2,
                                                                    random_split = True,
                                                                    sample_target = None
                                                                    )
    
    # initiate gbm
    gbm = ensemble.GradientBoostingRegressor(random_state = 123)
    
    # tune gbm model
    mod_tuning = va.tune_hyperparameters(model = gbm, 
                                         params = cons.train_age_gbm_params, 
                                         X_train = X_train, 
                                         y_train = y_train,
                                         scoring = 'neg_mean_squared_error'
                                         )
    
    # extract the best parameters
    best_params = mod_tuning.loc[0, 'params']
    
    # initiate the best model
    gbm = ensemble.GradientBoostingRegressor(learning_rate = best_params['learning_rate'],
                                             loss = best_params['loss'],
                                             max_depth = best_params['max_depth'],
                                             max_features = best_params['max_features'],
                                             n_estimators = best_params['n_estimators'],
                                             presort = best_params['presort'],
                                             random_state = 123
                                             )
    
    # fit the best model
    gbm.fit(X_train, 
            y_train.values.ravel()
            )
    
    # classify the validation set
    y_valid['Age_pred'] = gbm.predict(X_valid)
    
    # genrate the regression metrics
    va.perf_metrics(y_obs = y_valid['Age'], 
                    y_pred = y_valid['Age_pred'], 
                    target_type = 'reg'
                    )
    
    # create prediction, observation and residual plots
    va.Vis.preds_obs_resids(obs = 'Age',
                            preds = 'Age_pred',
                            dataset = y_valid
                            )
    
    # predict for the test set
    test['Age'] = gbm.predict(test[X_valid.columns])
    
    # re-concatenate the training and test to update the base train set
    base_train = pd.concat(objs = [train, test],
                           axis = 0
                           )

    print('Cleaning Age in base test ...')
    
    # randomly split the dataset
    X_valid, X_train, y_valid, y_train = va.train_test_split_sample(dataset = base_train,
                                                                    y = ['Age'],
                                                                    X = train.columns.drop(['Age', 'Survived', 'PassengerId', 'Dataset']),
                                                                    train_size = 0.8,
                                                                    test_size = 0.2,
                                                                    random_split = True,
                                                                    sample_target = None
                                                                    )
    
    # initiate gbm
    gbm = ensemble.GradientBoostingRegressor(random_state = 123)
    
    # tune gbm model
    mod_tuning = va.tune_hyperparameters(model = gbm, 
                                         params = cons.test_age_gbm_params, 
                                         X_train = X_train, 
                                         y_train = y_train,
                                         scoring = 'neg_mean_squared_error'
                                         )
    
    # extract the best parameters
    best_params = mod_tuning.loc[0, 'params']
    
    # initiate the best model
    gbm = ensemble.GradientBoostingRegressor(learning_rate = best_params['learning_rate'],
                                             loss = best_params['loss'],
                                             max_depth = best_params['max_depth'],
                                             max_features = best_params['max_features'],
                                             n_estimators = best_params['n_estimators'],
                                             presort = best_params['presort'],
                                             random_state = 123
                                             )
    
    # fit the best model
    gbm.fit(X_train, 
            y_train.values.ravel()
            )
    
    # classify the validation set
    y_valid['Age_pred'] = gbm.predict(X_valid)
    
    # genrate the regression metrics
    va.perf_metrics(y_obs = y_valid['Age'], 
                    y_pred = y_valid['Age_pred'], 
                    target_type = 'reg'
                    )
    
    # create prediction, observation and residual plots
    va.Vis.preds_obs_resids(obs = 'Age',
                            preds = 'Age_pred',
                            dataset = y_valid
                            )
    
    # predict for the base_test set
    base_test['Age'] = gbm.predict(base_test[X_valid.columns])
    
    # re-concatenate the base training and base test to update base data
    base = pd.concat(objs = [base_train, base_test],
                     axis = 0
                     )

    print('Outputting data ...')
    
    # output the dataset
    base.to_csv(base_clean_2_fpath,
                sep = '|',
                encoding = 'utf-8',
                header = True,
                index = False
                )
    
    return 0 

if __name__ == '__main__':
    
    clean_base_age(base_clean_fpath = cons.base_clean_data_fpath,
                   base_clean_2_fpath = cons.base_clean_2_data_fpath
                   )