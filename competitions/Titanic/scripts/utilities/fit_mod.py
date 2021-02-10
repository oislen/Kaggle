# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 15:10:47 2021

@author: oislen
"""

# load in relevant libraries
import pandas as pd
import cons
import value_analysis as va

def fit_mod(base_train,
            base_test,
            y_col,
            X_col,
            model,
            params,
            target_type = 'reg',
            random_state = 123,
            train_size = 0.8,
            test_size = 0.2,
            random_split = True,
            sample_target = None,
            sample_type = 'over',
            scoring = 'neg_mean_squared_error',
            cv = 10,
            n_jobs = -1,
            refit = True,
            return_mod = True,
            verbose = 0
            ):
    
    """
    
    Fit Model Documentation
    
    Function Overview
    
    This function fits sklearn model for the Titanic compeition
    
    Defaults
    
    fit_mod(base_train,
            base_test,
            y_col,
            X_col,
            model,
            params,
            random_state = 123,
            train_size = 0.8,
            test_size = 0.2,
            random_split = True,
            sample_target = None,
            scoring = 'neg_mean_squared_error',
            cv = 10,
            n_jobs = -1,
            refit = True,
            return_mod = True,
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
    sample_target - String, whether to sample the target attribute, default is None
    scoring - String, the type of scoring to perform on gbm model, default is 'neg_mean_squared_error'
    
    Returns
    
    base - DataFrame, the base data with filled age column
    
    Example
 
    fit_mod(base_train = train,
            base_test = test,
            y_col = ['Age'],
            X_col = ['Pclass', 'SibSp', 'Parch', 'FamSize', 'Fare', 'Alone', 'Mr', 'Mrs', 'Ms', 'Priv', 'male', 'Embarked'],
            params = cons.test_age_gbm_params,
            random_state = 123,
            train_size = 0.8,
            test_size = 0.2,
            random_split = True,
            sample_target = None,
            scoring = 'neg_mean_squared_error',
            cv = 10,
            n_jobs = -1,
            refit = True,
            return_mod = True,
            verbose = 0
            )
    
    """
    
    # extract out target
    tar_col = y_col[0]
    
    # create predicted column name
    pred_col = '{}_pred'.format(tar_col)
    
    # randomly split the dataset
    X_valid, X_train, y_valid, y_train = va.train_test_split_sample(dataset = base_train,
                                                                    y = y_col,
                                                                    X = X_col,
                                                                    train_size = train_size,
                                                                    test_size = test_size,
                                                                    random_split = random_split,
                                                                    sample_target = sample_target,
                                                                    sample_type = sample_type
                                                                    )
    
    # tune gbm model
    mod_tuning = va.tune_hyperparameters(model = model, 
                                         params = params, 
                                         X_train = X_train, 
                                         y_train = y_train,
                                         scoring = scoring,
                                         return_mod = return_mod,
                                         cv = cv,
                                         n_jobs = n_jobs,
                                         refit = refit,
                                         verbose = verbose
                                         )
    
    # if not refitting or retunring the best model
    if refit == False or return_mod == False:
        
        """
        # extract out the model tuning results
        mod_tuning_df = mod_tuning['tune_df']
        
        # extract the best parameters
        best_params = mod_tuning_df.loc[0, 'params']
        
        # initiate the best model
        gbm = ensemble.GradientBoostingRegressor(learning_rate = best_params['learning_rate'],
                                                 loss = best_params['loss'],
                                                 max_depth = best_params['max_depth'],
                                                 max_features = best_params['max_features'],
                                                 n_estimators = best_params['n_estimators'],
                                                 random_state = 123
                                                 )
        
        # fit the best model
        gbm.fit(X_train, 
                y_train.values.ravel()
                )
        """
       
        raise ValueError('wrong inputs')
        
    # otherwise
    else:
        
        # extract out the model of best fit
        gbm = mod_tuning['best_estimator']
        
    # classify the validation set
    y_valid[pred_col] = gbm.predict(X_valid)
    
    # genrate the regression metrics
    va.perf_metrics(y_obs = y_valid[tar_col], 
                    y_pred = y_valid[pred_col], 
                    target_type = target_type
                    )
    
    if target_type == 'class':
        
        # randomly split the dataset
        X_valid_emp, X_train, y_valid_emp, y_train = va.train_test_split_sample(dataset = base_train,
                                                                                y = y_col,
                                                                                X = X_col,
                                                                                train_size = 1,
                                                                                test_size = 0,
                                                                                random_split = False,
                                                                                sample_target = sample_target,
                                                                                sample_type = sample_type
                                                                                )
        
        # refit model to all training data
        gbm.fit(X_train[X_col], 
                y_train[y_col].values.ravel()
                )
        
    else :
        
        # refit model to all training data
        gbm.fit(base_train[X_col], 
                base_train[y_col].values.ravel()
                )

    # predict for the base_test set
    base_test[tar_col] = gbm.predict(base_test[X_col])
    
    # if running regression
    if target_type == 'reg':
            
        # create prediction, observation and residual plots
        va.Vis.preds_obs_resids(obs = tar_col,
                                preds = pred_col,
                                dataset = y_valid
                                )
        
        # plot predicted age
        va.Vis.hist(dataset = y_valid,
                    num_var = [pred_col],
                    title = 'Histogram of Predicted {} - Validation Set'.format(tar_col)
                    )
        
        # plot predicted age
        va.Vis.hist(dataset = base_test,
                    num_var = [tar_col],
                    title = 'Histogram of Predicted {} - Test Set'.format(tar_col)
                    )
    
    # otherwise if classification
    elif target_type == 'class':
        
        # create a ROC curve
        va.Vis.roc_curve(obs = tar_col, 
                         preds = pred_col, 
                         dataset = y_valid
                         )
    
    # re-concatenate the base training and base test to update base data
    base = pd.concat(objs = [base_train, base_test],
                     axis = 0
                     )
    
    return base
