# -*- coding: utf-8 -*-
"""
Created on Fri May 21 13:11:43 2021

@author: oislen
"""

import numpy as np
from sklearn.model_selection import cross_val_score

def rmse_cv(model, 
            X_train, 
            y,
            scoring = 'neg_mean_squared_error',
            cv = 5
            ):
    
    """
    
    Root Mean Squared Error Documentation
    
    Function Overview
    
    This function generates the rmse metric for a model being validated with cross validation
    
    Defaults
    
    rmse_cv(model, 
            X_train, 
            y,
            scoring="neg_mean_squared_error",
            cv = 5
            )
    
    Parameters
    
    model - Sklearn  Model, the model being trained
    X_train - DataFrame, the training data
    y - Series, the target variable
    scoring - String, the type of scoring to use, default is 'neg_mean_squared_error'
    cv - Integerm the number of cv validaiton folds
    
    Returns
    
    rmse - array, the cross validation results
    
    Example
    
    rmse_cv(mode = model, 
            X_train = X_train, 
            y = y,
            scoring = 'neg_mean_squared_error',
            cv = 5
            )
    
    """
    
    # create mean square error metric
    metric = -cross_val_score(model, X_train, y, scoring = scoring, cv = cv)
    
    # convert to root mean squared error 
    rmse = np.sqrt(metric)
    
    return rmse