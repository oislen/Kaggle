# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 15:10:47 2021

@author: oislen
"""

# load in relevant libraries
import cons
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline
from model.perf_metrics import perf_metrics
from graph.hist import hist
from graph.vis_feat_imp import vis_feat_imp
from graph.learning_curve import learning_curve
from graph.roc_curve import roc_curve

def fit_sur_mod(base_train,
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
                scoring = 'accuracy',
                cv = 10,
                n_jobs = -1,
                refit = True,
                verbose = 0
                ):
    
    """
    
    Fit Survival Model Documentation
    
    Function Overview
    
    This function fits an sklearn model for the Titanic compeition.
    The process includes splitting the training data into training and validation (holdout) sets.
    SMOTE is applied to syntetically up sample the minor class of survived.
    Grid search cross validation is then applied to find the optimal parameters for the model.
    Once the optimal model is found, the model is validated using the validation (holdout) set.
    Learning curves, performance metrics and ROC curves are all use to evaluate the final model.
    The final model is then refitted to the entire training set and predictions are made for the test set.
    The final model and its predictions are saved for reproduceability and ensemble modelling.
    
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
    model_name - String, the name of the model to run, see model parameters in cons.py
    model - Sklearn Model, the model to fit, see model definition in cons.py
    params - Dictionary, the gbm model parameters to tune
    random_state - Integer, the random seed to set, default is 123
    train_size - Float, the proportion of data to have in training set, default is 0.8
    test_size - Float, the proportion of data to have in the testing set, default is 0.2
    random_split - Boolean, whether to randomise the data before splitting, default is True
    scoring - String, the type of scoring to perform on gbm model, default is 'accuracy'
    cv - Integer, the number of folds to use for cross fold validation when training the model, default is 10
    n_jobs - Integer, the number of cores to use when processing data, default is -1 for all cores
    refit - Boolean, whether to refit the best model following grid search cross validation hypter parameter tuning, default is True
    verbose - Integer, whether to print verbose updates when tuning model, default is 0
    
    Returns
    
    base - DataFrame, the base data with filled age column
    
    Example
 
    fit_sur_mod(base_train = train,
                base_test = test,
                y_col = ['Survived'],
                X_col = ['Pclass', 'SibSp', 'Parch', 'FamSize', 'Fare', 'Alone', 'Mr', 'Mrs', 'Ms', 'Priv', 'male', 'Embarked'],
                params = cons.test_age_gbm_params,
                random_state = 123,
                train_size = 0.8,
                test_size = 0.2,
                random_split = True,
                scoring = 'accuracy',
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
    
    # create count plot of classifications
    hist(dataset = base_train,
         num_var = [tar_col],
         output_dir = cons.model_results_dir.format(model_name),
         output_fname = cons.hist_train_tar_fname.format(model_name)
         )
    
    print('splitting data into training and validation sets ...')
    
    # split the training data
    X_train, X_valid, y_train, y_valid = train_test_split(base_train[X_col], 
                                                          base_train[y_col], 
                                                          train_size = train_size,
                                                          test_size = test_size, 
                                                          shuffle = random_split,
                                                          random_state = cons.random_state
                                                          )

    print('running hyperparameter tuning ...')
    
    # create define a smote object
    smote = SMOTE(random_state = cons.random_state)
    
    # create pipeline with smote and model
    imba_pipeline = make_pipeline(smote, model)
    
    # extract out the full model name from the pipeline
    full_name = imba_pipeline.steps[1][0]
    
    # update the parameter dictionary with the full model name
    for key in list(params.keys()):
        params['{}__{}'.format(full_name, key)] = params.pop(key)
        
    # create grid search cross validation object
    mod_tuning = GridSearchCV(estimator = imba_pipeline,
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
    best_estimator = mod_tuning.best_estimator_
    best_model = best_estimator.named_steps[full_name]
    best_params = mod_tuning.best_params_
    best_score = mod_tuning.best_score_
    
    # print best parameters and best score
    print(best_params)
    print(best_score)
    
    # create learning curve
    learning_curve(model = best_model,
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
       
    print('predicting for validation set ...')
    
    # classify the validation set
    y_valid[pred_col] = best_model.predict(X_valid)
    
    print('evaluating validation predictions ...')
    
    # create count plot of classifications
    hist(dataset = y_valid,
         num_var = [pred_col],
         output_dir = cons.model_results_dir.format(model_name),
         output_fname = cons.hist_valid_preds_fname.format(model_name)
         )

    # genrate the regression metrics
    val_metrics = perf_metrics(y_obs = y_valid[tar_col], 
                               y_pred = y_valid[pred_col], 
                               target_type = 'class',
                               output_dir = cons.model_results_dir.format(model_name),
                               output_fname = cons.metrics_fname.format(model_name)
                               )
    
    # print the validation metrics
    print(val_metrics)
    
    # create a ROC curve
    roc_curve(obs = tar_col, 
              preds = pred_col, 
              dataset = y_valid,
              output_dir = cons.model_results_dir.format(model_name),
              output_fname = cons.roc_fname.format(model_name)
              )

    print('refitting to all training data ...')

    final_model = make_pipeline(smote, best_model)
    
    # refit model to all training data
    final_model.fit(base_train[X_col], 
                    base_train[y_col[0]]
                    )

    # pickle the best model
    joblib.dump(final_model, cons.best_model_fpath.format(model_name))

    # predict for the base_test set
    base_test[tar_col] = best_model.predict(base_test[X_col])
    
    # create count plot of classifications
    hist(dataset = base_test,
         num_var = [tar_col],
         output_dir = cons.model_results_dir.format(model_name),
         output_fname = cons.hist_test_preds_fname.format(model_name)
         )
    
    return base_test
