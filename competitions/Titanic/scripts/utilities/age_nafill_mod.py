# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 15:10:47 2021

@author: oislen
"""

# load in relevant libraries
import pandas as pd
from sklearn import ensemble
import value_analysis as va

def age_nafill_mod(base_train,
                   base_test,
                   y_col,
                   X_col,
                   params,
                   random_state = 123,
                   train_size = 0.8,
                   test_size = 0.2,
                   random_split = True,
                   sample_target = None,
                   scoring = 'neg_mean_squared_error'
                   ):
    
    """
    
    Age NA Fill Model Documentation
    
    Function Overview
    
    This function fills in the missing age values for the Titanic base data
    
    Defaults
    
    age_nafill_mod(base_train,
                   base_test,
                   y_col,
                   X_col,
                   params,
                   random_state = 123,
                   train_size = 0.8,
                   test_size = 0.2,
                   random_split = True,
                   sample_target = None,
                   scoring = 'neg_mean_squared_error'
                   )
    
    Parameters
    
    base_train - DataFrame, the base training data
    base_test - DataFrame, the base testing data
    y_col - List of Strings, the target y column
    X_col - List of Strings, the predictor X columns
    params - Dictionary, the gbm model parameters to tune
    random_state - Integer, the random seed to set, default is 123
    train_size - Float, the proportion of data to have in training set, default is 0.8
    test_size - Float, the proportion of data to have in the testing set, default is 0.2
    random_split - Boolean, whether to randomise the data before splitting, default is True
    sample_target - String, whether to sample the target attribute, default is None
    scoring - String, the type of scoring to perform on gbm model, default is 'neg_mean_squared_error'
    
    Returns
    
    base - DataFrame, the base data with filled age column
    
    Example
    
    age_nafill_mod(base_train = train,
                   base_test = test,
                   y_col = ['Age'],
                   X_col = ['Pclass', 'SibSp', 'Parch', 'FamSize', 'Fare', 'Alone', 'Mr', 'Mrs', 'Ms', 'Priv', 'male', 'Embarked'],
                   params = cons.test_age_gbm_params,
                   random_state = 123,
                   train_size = 0.8,
                   test_size = 0.2,
                   random_split = True,
                   sample_target = None,
                   scoring = 'neg_mean_squared_error'
                   )
        
    """
    
    # randomly split the dataset
    X_valid, X_train, y_valid, y_train = va.train_test_split_sample(dataset = base_train,
                                                                    y = y_col,
                                                                    X = X_col,
                                                                    train_size = train_size,
                                                                    test_size = test_size,
                                                                    random_split = random_split,
                                                                    sample_target = sample_target
                                                                    )
    
    # initiate gbm
    gbm = ensemble.GradientBoostingRegressor(random_state = random_state)
    
    # tune gbm model
    mod_tuning = va.tune_hyperparameters(model = gbm, 
                                         params = params, 
                                         X_train = X_train, 
                                         y_train = y_train,
                                         scoring = scoring
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
    
    return base
