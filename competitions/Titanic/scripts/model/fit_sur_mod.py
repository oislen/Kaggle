# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 15:10:47 2021

@author: oislen
"""

# load in relevant libraries
import pandas as pd
import cons
import value_analysis as va
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
    
    
    # create count plot of classifications
    va.Vis.hist(dataset = base_train,
                 num_var = [tar_col],
                 output_dir = cons.model_results_dir.format(model_name),
                 output_fname = cons.hist_train_tar_fname.format(model_name)
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
    
    # save the tuning results
    mod_tuning_df.to_csv(cons.hyper_param_fpath.format(model_name, model_name),
                         index = False
                         )

    # extract out the model of best fit
    best_model = mod_tuning['best_estimator']
    
    # pickle the best model
    joblib.dump(best_model, cons.best_model_fpath.format(model_name, model_name))
    
    # create learning curve
    va.Vis.learning_curve(model = best_model,
                          X_train = X_train,
                          y_train = y_train,
                          scoring = 'accuracy',
                          title = 'Learning Curve: {}'.format(model_name.upper()),
                          output_dir = cons.model_results_dir.format(model_name),
                          output_fname = cons.learning_curve_fnamt.format(model_name)
                          )
    
    if model_name in ['rfc', 'abc', 'etc', 'gbc']:
        
        # plot feature importance
        vis_feat_imp(name = model_name, model = best_model, X_train = X_train)
       
    # if there is a spare validation set
    if y_valid.shape[0] > 0:
            
        print('predicting for validation set ...')
        
        # classify the validation set
        y_valid[pred_col] = best_model.predict(X_valid)
        
        print('evaluating validation predictions ...')
        
        # create count plot of classifications
        va.Vis.hist(dataset = y_valid,
                    num_var = [pred_col],
                    output_dir = cons.model_results_dir.format(model_name),
                    output_fname = cons.hist_valid_preds_fname.format(model_name)
                    )
    
        # genrate the regression metrics
        val_metrics = va.perf_metrics(y_obs = y_valid[tar_col], 
                                      y_pred = y_valid[pred_col], 
                                      target_type = target_type,
                                      output_dir = cons.model_results_dir.format(model_name),
                                      output_fname = cons.metrics_fname.format(model_name)
                                      )
        
        # print the validation metrics
        print(val_metrics)
        
        # create a ROC curve
        va.Vis.roc_curve(obs = tar_col, 
                         preds = pred_col, 
                         dataset = y_valid,
                         output_dir = cons.model_results_dir.format(model_name),
                         output_fname = cons.roc_fname.format(model_name)
                         )
        
    print('refitting to all training data ...')
    
    # refit model to all training data
    best_model.fit(base_train[X_col], 
                   base_train[y_col].values.ravel()
                   )

    print('predicting for test set ...')

    # predict for the base_test set
    base_test[tar_col] = best_model.predict(base_test[X_col])
    
    # create count plot of classifications
    va.Vis.hist(dataset = base_test,
                num_var = [tar_col],
                output_dir = cons.model_results_dir.format(model_name),
                output_fname = cons.hist_test_preds_fname.format(model_name)
                )
    
    # re-concatenate the base training and base test to update base data
    base = pd.concat(objs = [base_train, base_test],
                     axis = 0
                     )
    
    return base