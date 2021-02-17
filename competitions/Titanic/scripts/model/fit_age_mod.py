# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 15:10:47 2021

@author: oislen
"""

# load in relevant libraries
import pandas as pd
import cons
from sklearn.model_selection import train_test_split, GridSearchCV
from model.perf_metrics import perf_metrics
from graph.preds_obs_resids import preds_obs_resids
from graph.hist import hist

def fit_age_mod(base_train,
                base_test,
                y_col,
                X_col,
                model,
                params,
                random_state = 123,
                train_size = 0.8,
                test_size = 0.2,
                random_split = True,
                scoring = 'neg_mean_squared_error',
                cv = 10,
                n_jobs = -1,
                refit = True,
                verbose = 0
                ):
    
    """
    
    Fit Age Model Documentation
    
    Function Overview
    
    This function fits an age model for the Titanic compeition in order to impute missing values.
    The process includes splitting the training data into training and validation (holdout) sets.
    Grid search cross validation is then applied to find the optimal parameters for the model.
    Once the optimal model is found, the model is validated using the validation (holdout) set.
    Performance metrics and residual plots are all use to evaluate the final model.
    The final model is then refitted to the entire training set and predictions are made for the test set.
    
    Defaults
    
    fit_age_mod(base_train,
                base_test,
                y_col,
                X_col,
                model,
                params,
                random_state = 123,
                train_size = 0.8,
                test_size = 0.2,
                random_split = True,
                scoring = 'neg_mean_squared_error',
                cv = 10,
                n_jobs = -1,
                refit = True,
                verbose = 0
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
    scoring - String, the type of scoring to perform on gbm model, default is 'neg_mean_squared_error'
    cv - Integer, the number of folds to use for cross fold validation when training the model, default is 10
    n_jobs - Integer, the number of cores to use when processing data, default is -1 for all cores
    refit - Boolean, whether to refit the best model following grid search cross validation hypter parameter tuning, default is True
    verbose - Integer, whether to print verbose updates when tuning model, default is 0
    
    Returns
    
    base - DataFrame, the base data with filled age column
    
    Example
 
    fit_age_mod(base_train = train,
                base_test = test,
                y_col = ['Age'],
                X_col = ['Pclass', 'SibSp', 'Parch', 'FamSize', 'Fare', 'Alone', 'Mr', 'Mrs', 'Ms', 'Priv', 'male', 'Embarked'],
                params = cons.test_age_gbm_params,
                random_state = 123,
                train_size = 0.8,
                test_size = 0.2,
                random_split = True,
                scoring = 'neg_mean_squared_error',
                cv = 10,
                n_jobs = -1,
                refit = True,
                verbose = 0
                )
    
    """
    
    # extract out target
    tar_col = y_col[0]
    
    # create predicted column name
    pred_col = '{}_pred'.format(tar_col)
    
    # split the training data
    X_train, X_valid, y_train, y_valid = train_test_split(base_train[X_col], 
                                                          base_train[y_col], 
                                                          train_size = train_size,
                                                          test_size = test_size, 
                                                          shuffle = random_split,
                                                          random_state = cons.random_state
                                                          )

    # create grid search cross validation object
    mod_tuning = GridSearchCV(estimator = model,
                              param_grid = params, 
                              cv = cv,
                              scoring = scoring, 
                              n_jobs = n_jobs, 
                              refit = refit,
                              verbose = verbose
                              )
    
    # tune model
    mod_tuning.fit(X_train, y_train[y_col[0]])
    
    # extract out the model of best fit
    model = mod_tuning.best_estimator_
    best_params = mod_tuning.best_params_
    best_score = mod_tuning.best_score_
    
    # print tuning results
    print(best_params)
    print(best_score)
    
    # classify the validation set
    y_valid[pred_col] = model.predict(X_valid)
    
    # genrate the regression metrics
    reg_perf_metrics = perf_metrics(y_obs = y_valid[tar_col], 
                                    y_pred = y_valid[pred_col], 
                                    target_type = 'reg'
                                    )
    
    # output performance metrics
    print(reg_perf_metrics)
    
    # refit model to all training data
    model.fit(base_train[X_col], 
              base_train[y_col].values.ravel()
              )

    # predict for the base_test set
    base_test[tar_col] = model.predict(base_test[X_col])
    
    # create prediction, observation and residual plots
    preds_obs_resids(obs = tar_col,
                     preds = pred_col,
                     dataset = y_valid
                     )
    
    # plot predicted age
    hist(dataset = y_valid,
         num_var = [pred_col],
         title = 'Histogram of Predicted {} - Validation Set'.format(tar_col)
         )
    
    # plot predicted age
    hist(dataset = base_test,
         num_var = [tar_col],
         title = 'Histogram of Predicted {} - Test Set'.format(tar_col)
         )

    # re-concatenate the base training and base test to update base data
    base = pd.concat(objs = [base_train, base_test],
                     axis = 0
                     )
    
    return base
