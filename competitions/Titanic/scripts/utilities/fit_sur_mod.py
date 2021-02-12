# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 15:10:47 2021

@author: oislen
"""

# load in relevant libraries
import pandas as pd
import cons
import value_analysis as va
import os
import joblib
from vis_feat_imp import vis_feat_imp

def fit_sur_mod(base_train,
                base_test,
                y_col,
                X_col,
                model_name,
                model,
                params,
                target_type = 'class',
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
                report_dir = None,
                verbose = 0
                ):
    
    """
    
    Fit Model Documentation
    
    Function Overview
    
    This function fits sklearn model for the Titanic compeition
    
    Defaults
    
    fit_sur_mod(base_train,
                base_test,
                y_col,
                X_col,
                model_name,
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
 
    fit_sur_mod(base_train = train,
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
    
    # define the histogram of valid predictions filename
    hist_train_tar_fname = '{}_hist_train_tar.png'.format(model_name)
    
    # create count plot of classifications
    va.Vis.hist(dataset = base_train,
                 num_var = [tar_col],
                 output_dir = report_dir,
                 output_fname = hist_train_tar_fname
                 )
    
    print('splitting data into training and validation sets ...')
    
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
    
    print('running hyperparameter tuning ...')
    
    # tune gbm model
    mod_tuning = va.tune_hyperparameters(model = model, 
                                         params = params, 
                                         X_train = X_train, 
                                         y_train = y_train,
                                         scoring = scoring,
                                         cv = cv,
                                         n_jobs = n_jobs,
                                         refit = refit,
                                         verbose = verbose
                                         )

    # extract out the model tuning results
    mod_tuning_df = mod_tuning['tune_df']
    
    # define the filename
    hyperparam_fname = '{}_hyperparam_tuning.csv'.format(model_name)
    
    # definte the output file path
    hyper_param_fpath = os.path.join(report_dir, hyperparam_fname)
    
    # save the tuning results
    mod_tuning_df.to_csv(hyper_param_fpath,
                         index = False
                         )

    # extract out the model of best fit
    best_model = mod_tuning['best_estimator']
    
    # define the filename
    best_model_fname = '{}_best_model.pkl'.format(model_name)
    
    # model file path
    best_model_fpath = os.path.join(report_dir, best_model_fname)
    
    # pickle the best model
    joblib.dump(best_model, best_model_fpath)
    
    # definte the filename
    learning_curve_fnamt = '{}_learning_curve.png'.format(model_name)
    
    # create learning curve
    va.Vis.learning_curve(model = best_model,
                          X_train = X_train,
                          y_train = y_train,
                          scoring = 'accuracy',
                          title = 'Learning Curve: {}'.format(model_name.upper()),
                          output_dir = report_dir,
                          output_fname = learning_curve_fnamt
                          )
    
    if model_name in ['rfc', 'abc', 'etc', 'gbc']:
        
        # plot feature importance
        vis_feat_imp(name = model_name, model = best_model, X_train = X_train)
        
    print('predicting for validation set ...')
    
    # classify the validation set
    y_valid[pred_col] = best_model.predict(X_valid)
    
    print('evaluating validation predictions ...')
    
    # define the histogram of valid predictions filename
    hist_valid_preds_fname = '{}_hist_valid_preds.png'.format(model_name)
    
    # create count plot of classifications
    va.Vis.hist(dataset = y_valid,
                num_var = [pred_col],
                output_dir = report_dir,
                output_fname = hist_valid_preds_fname
                )
    
    # definte the metrics filename
    metrics_fname = '{}_perf_metrics.csv'.format(model_name)
    
    # genrate the regression metrics
    val_metrics = va.perf_metrics(y_obs = y_valid[tar_col], 
                                  y_pred = y_valid[pred_col], 
                                  target_type = target_type,
                                  output_dir = report_dir,
                                  output_fname = metrics_fname
                                  )
    
    # print the validation metrics
    print(val_metrics)
    
    # define the roc filename
    roc_fname = '{}_roc_curve.png'.format(model_name)
    
    # create a ROC curve
    va.Vis.roc_curve(obs = tar_col, 
                     preds = pred_col, 
                     dataset = y_valid,
                     output_dir = report_dir,
                     output_fname = roc_fname
                     )
    
    print('refitting to all training data ...')
    
    # refit model to all training data
    best_model.fit(base_train[X_col], 
                   base_train[y_col].values.ravel()
                   )

    print('predicting for test set ...')

    # predict for the base_test set
    base_test[tar_col] = best_model.predict(base_test[X_col])
    
    # define the histogram of valid predictions filename
    hist_test_preds_fname = '{}_hist_test_preds.png'.format(model_name)
    
    # create count plot of classifications
    va.Vis.hist(dataset = base_test,
                num_var = [tar_col],
                output_dir = report_dir,
                output_fname = hist_test_preds_fname
                )
    
    # re-concatenate the base training and base test to update base data
    base = pd.concat(objs = [base_train, base_test],
                     axis = 0
                     )
    
    return base
