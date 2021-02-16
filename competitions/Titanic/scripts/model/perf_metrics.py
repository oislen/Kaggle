# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 16:45:15 2021

@author: oislen
"""

#####################
#-- Preliminaries --#
#####################

# import the relevant libraries
import pandas as pd
import numpy as np
import sklearn.metrics as metrics
from sklearn.model_selection import KFold, cross_val_score

def perf_metrics(y_obs = None,
                 y_pred = None,
                 model = None,
                 endog = None,
                 exog = None,
                 target_type = 'class',
                 n_folds = None,
                 digits = 3,
                 output_dir = None,
                 output_fname = None
                 ):
    
    """
    
    Metrics Documentation
    
    Function Overview
    
    This function produces a variety of classification and regression metrics for given observations and predictions.
    The classification metrics include accuracy, precision, recall, auc, f1, hamming, jaccard similarity and zero one erro.
    The regression metrics include mean squared error, mean absolute errerr, median absolute error, mean squared log error, correlation and variance explained.
    The function can also perform k-folds cross-validation given a model, endog, exog and a specified number of folds.
    
    Defaults
    
    perf_metrics(y_obs,
                 y_pred,
                 target_type = 'class',
                 model = None,
                 endog = None,
                 exog = None,
                 n_folds = None,
                 output_dir = None,
                 output_fname = None
                 )
    
    Parametres
    
    y_obs - pd.Series, a pandas series of the observed values
    y_pred - pd.Series, a pandas series of predicted values
    model - Sklearn Model, a model evaluate with cross-fold validation, default is None.
    endog - Series, the target data to perform the cross validation on, default is None.
    exog - DataFrame, the training data to perform the cross validation on, default is None.
    target_type - String, the type of prediction, either 'reg' for regression or 'class' for classification. 
    n_folds - Integer, the number of folds for validating on, default is 5.
    digits - Integer, the number of digist to round the output to, default is 3.
    output_dir - String, the output directory of the feature importance, default None.
    output_fname - String, the filename and extension of the output file, default is None.
    
    Returns
    
    metrics - DataFrame, the cross-fold validation metrics
    
    Example
    
    perf_metrics(y_obs = losscost, 
                 y_pred = pred_cancellation, 
                 target_type = 'class',
                 output_dir = '/opt/dataprojects/Analysis',
                 output_fname = 'metrics_logistic.csv
                 )
    
    See Also
    
    GAM_feat_imp, GLM.simple_analysis, MARS_feat_imp
    
    References
    
    https://scikit-learn.org/stable/modules/model_evaluation.html
    
    """
    
    ######################
    #-- Error Handling --#
    ######################
    
    # error handling for 'target'
    if target_type not in ['class', 'reg']:
        
        # print error message and end script
        return "Error: incorrect target_type given, must be either 'class' or 'reg'"
    
    ###############
    #-- K-Folds --#
    ###############
    
    # if doing k-folds cross-validation
    if n_folds != None:
        
        # create the k-folds
        kf = KFold(n_folds, shuffle = True, random_state = 1234)
        
        # generate the splits in the training data
        kf = kf.get_n_splits(exog.values)
    
    ##############################
    #-- Classification Metrics --#
    ##############################
    
    # if the predictions come from a classification model
    if target_type == 'class':
    
        # define the output statistics
        metric_names = ['Acc', 'Prec', 'Recall', 'AUC', 'F1']
        
        #-- Straight Metrics --#
        
        # if the y observations and predictions are given
        if (y_obs is not None) and (y_pred is not None):
        
            # calculate the metrics
            Acc = metrics.accuracy_score(y_true = y_obs, y_pred = y_pred)
            Prec = metrics.precision_score(y_true = y_obs, y_pred = y_pred)
            Recall = metrics.recall_score(y_true = y_obs, y_pred = y_pred)
            AUC = metrics.roc_auc_score(y_true = y_obs, y_score  = y_pred)
            F1 = metrics.f1_score(y_true = y_obs, y_pred = y_pred)

        #-- Cross Validation --#
        
        # else if a model is given with data for cross-validation
        elif (model is not None) and (exog is not None) and (endog is not None):

            # calculate the metrics via cross fold validation
            Acc = cross_val_score(model, exog.values, endog, scoring="accuracy", cv = kf)
            Prec = cross_val_score(model, exog.values, endog, scoring="precision", cv = kf)
            Recall = cross_val_score(model, exog.values, endog, scoring="recall", cv = kf)
            AUC = cross_val_score(model, exog.values, endog, scoring="roc_auc", cv = kf)
            F1 = cross_val_score(model, exog.values, endog, scoring="f1", cv = kf)
    
        # define a list of output metrics
        metric_outputs = [Acc, Prec, Recall, AUC, F1]
      
    ##########################
    #-- Regression Metrics --#
    ##########################

    # if the predictions come from a regression model
    elif target_type == 'reg':
    
        # define the output statistics
        metric_names = ['MSE', 'MAE', 'MdAE', 'R2', 'Exp Var']

        #-- Straight Metrics --#

        # if the y observations and predictions are given
        if (y_obs is not None) and (y_pred is not None):
        
            # calculate the metrics via cross fold validation
            MSE = metrics.mean_squared_error(y_true = y_obs, y_pred = y_pred)
            MAE = metrics.mean_absolute_error(y_true = y_obs, y_pred = y_pred)
            MdAE = metrics.median_absolute_error(y_true = y_obs, y_pred = y_pred)
            R2 = metrics.r2_score(y_true = y_obs, y_pred = y_pred)
            Exp_Var = metrics.explained_variance_score(y_true = y_obs, y_pred = y_pred)

        #-- Cross Validation --#
        
        # else if a model is given with data for cross-validation
        elif (model is not None) and (exog is not None) and (endog is not None):
  
            # calculate the metrics via cross fold validation
            MSE = -cross_val_score(model, exog.values, endog, scoring="neg_mean_squared_error", cv = kf)
            MAE = -cross_val_score(model, exog.values, endog, scoring="neg_mean_absolute_error", cv = kf)
            MdAE = -cross_val_score(model, exog.values, endog, scoring="neg_median_absolute_error", cv = kf)
            R2 = cross_val_score(model, exog.values, endog, scoring="r2", cv = kf)
            Exp_Var = cross_val_score(model, exog.values, endog, scoring="explained_variance", cv = kf)
        
        # define a list of output metrics
        metric_outputs = [MSE, MAE, MdAE, R2, Exp_Var]
        
    ##############
    #-- Output --#
    ##############
    
    # create a dictionary for the output dataframe
    metric_dict = dict(zip(metric_names, metric_outputs))
    
    # if not performing k-folds cv
    if n_folds == None:
        
        # create the index of the output dataframe
        idx = np.arange(1)
        
    # else if performing k-folds cv
    elif n_folds != None:
        
        # create the index of the output dataframe
        idx = np.arange(n_folds)
    
    # create the output dataframe
    metric_df = pd.DataFrame(metric_dict, index = idx)
    
    # round the output to the specified number of digits
    metric_df = metric_df.round(decimals = digits)
    
    # if the output directory is given
    if output_dir != None:
        
        # if the output filename is given
        if output_fname != None:
            
            # set the filename
            filename = output_fname
            
        # else if the output filename is not given
        elif output_fname == None:
            
            # create the filename
            filename = 'metrics_model.csv'
        
        # create the absolute file path
        abs_file_path = output_dir + '/' + filename
        
        # save the feature importance
        metric_df.to_csv(path_or_buf = abs_file_path,
                         sep = '|',
                         header = True,
                         index = True,
                         encoding = 'latin1'
                         )
        
    # return the metrics dataframe
    return metric_df

