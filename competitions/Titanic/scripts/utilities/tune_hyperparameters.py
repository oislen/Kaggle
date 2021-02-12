# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 16:40:10 2021

@author: oislen
"""

#####################
#-- Preliminaries --#
#####################

# import the relevant libraries
import pandas as pd
from sklearn.model_selection import GridSearchCV

############################
#-- Tune Hyperparameters --#
############################

def tune_hyperparameters(model, 
                         params,
                         X_train,
                         y_train,
                         scoring = 'accuracy',
                         cv = 10,
                         n_jobs = -1,
                         refit = True,
                         verbose = 0
                         ):
    
    """
    
    Tune Hyperparameters Documentation
    
    Function Overview
    
    Create a function to tune hyperparameters of the selected models.
    
    Default
    
    tune_hyperparameters(model, 
                         params,
                         X_train,
                         y_train,
                         scoring = 'accuracy'
                         )
    
    Parameters
    
    model - Sklearn Classifier or Regressor
    params - Dictionary, the parameters to tune
    X_trian - Dataframe, the predictors of the training data
    y_train - Dataframe, the respones of the training data
    scoring - String, the metric used in cross validation
    
    Returns
    
    tune_df - Dataframe, the hyperparameter tuning and cross-validation results.
    
    Example
    
    tune_hyperparameters(model = LogisticRegression(), 
                         params = lr_params,
                         X_train = X_train,
                         y_train = y_train,
                         scoring = 'accuracy'
                         )
    
    See Also
    
    metrics, vis_roc_curve 
    
    References
    
    https://www.kaggle.com/eraaz1/a-comprehensive-guide-to-titanic-machine-learning
    
    """
    
    # Construct grid search object with 10 fold cross validation.
    grid = GridSearchCV(model, 
                        param_grid  = params, 
                        verbose = verbose, 
                        cv = cv, 
                        scoring = scoring,
                        n_jobs = n_jobs,
                        refit = refit
                        )
    
    # Fit using grid search.
    grid.fit(X_train, y_train.values.ravel())
    
    # extract out the relevant information
    params = grid.cv_results_['params']
    
    # extract test score stats
    mean_test_score = grid.cv_results_['mean_test_score']
    std_test_score = grid.cv_results_['std_test_score']
    rank_test_score = grid.cv_results_['rank_test_score']
    
    # create a dictionary to hold the data
    tune_dict = {'params':params,
                 'std_test_score':std_test_score,
                 'mean_test_score':mean_test_score,
                 'rank_test_score':rank_test_score
                 }
    
    # set the split name
    split_base = 'split{}_test_score'
    
    # loop over each split
    for split in range(cv):
        
        # create full splut name
        split_name = split_base.format(split)
    
        # extract the test scores
        split_test_score = grid.cv_results_[split_name]
        
        # assign split score to dictionary
        tune_dict[split_name] = split_test_score
        
    # create a dataframe to hold the data
    tune_df = pd.DataFrame(tune_dict).sort_values(['rank_test_score'],
                                                  ascending  = True
                                                  )
    # create output dictionary
    out_dict = {}
    out_dict['tune_df'] = tune_df
    
    # return the dataframe
    if refit == True:
        
        # extract out best model
        best_estimator = grid.best_estimator_ 
        
        # extract out best score
        best_score = grid.best_score_  
        
        # assign best model to output dict 
        out_dict['best_estimator'] = best_estimator
        out_dict['best_score'] = best_score
        
    return out_dict
