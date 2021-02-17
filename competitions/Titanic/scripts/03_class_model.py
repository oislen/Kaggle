# -*- coding: utf-8 -*-
"""
Created on Sun Nov  4 22:19:35 2018

@author: oislen
"""

# load in relevant libraries
import pandas as pd
from model.fit_sur_mod import fit_sur_mod
import cons
import os

def class_model(base_engin_fpath,
                pred_fpath,
                random_state = 123,
                train_size = 0.8,
                test_size = 0.3,
                random_split = True,
                scoring = 'accuracy',
                cv = 10,
                n_jobs = -1,
                refit = True,
                verbose = 0
                ):
    
    """
    
    Classification GBM Documentation
    
    Function Overview
    
    This function iterates over the various sklearn models and trainings each one using the fit_sur_model() function.
    
    
    Defaults
    
    class_model(base_engin_fpath,
                pred_fpath,
                random_state = 123,
                train_size = 0.8,
                test_size = 0.3,
                random_split = True,
                scoring = 'accuracy',
                cv = 10,
                n_jobs = -1,
                refit = True,
                verbose = 0
                )
    
    Parameters
    
    base_engin_fpath - String, the input file path for the base engineered data
    pred_fpath - String, the output file path for the GBM classifications
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
    
    0 for successful execution
    
    Example
    
    class_model(base_engin_fpath = 'C:\\Users\\...\\base_engin.csv',
                pred_fpath = 'C:\\Users\\...\\preds.csv',
                random_state = 123,
                train_size = 0.8,
                test_size = 0.3,
                random_split = True,
                scoring = 'accuracy',
                cv = 10,
                n_jobs = -1,
                refit = True,
                verbose = 0
                )
        
    """
    
    print('Checking inputs ...')
    
    # check input data types
    str_inputs = [base_engin_fpath, pred_fpath]
    if any([type(val) != str for val in str_inputs]):
        raise ValueError('Input params [base_engin_fpath, pred_fpath] must be str data types')
    # check if input file path exists
    if os.path.exists(base_engin_fpath) == False:
        raise OSError('Input file path {} does not exist'.format(base_engin_fpath))
    
    print('Loading engineered feature data ...')
    
    # load in data
    base = pd.read_csv(base_engin_fpath, 
                       sep = cons.sep
                       )
    
    print('Splitting data into training and test sets ...')
    
    # split the data based on the original dataset
    base_train = base[base.Dataset == 'train']
    base_test = base[base.Dataset == 'test']
    
    # extract out model and params
    sur_dict = cons.sur_dict
    
    # iterate over each model and evaluate it
    for model_name in sur_dict.keys():
        
        print('~~~~~ Working on model: {} ...'.format(model_name))
        
        # set model constants gien model name
        y_col = cons.y_col
        X_col =  base.columns.drop(cons.id_cols).tolist()
        model = sur_dict[model_name]['model']
        params = sur_dict[model_name]['params']
        report_dir = cons.model_results_dir.format(model_name)
        
        # check if model report directory exists
        if os.path.exists(report_dir) == False:
            # if not, create it
            os.makedirs(report_dir)
        
        # fit model
        base_test = fit_sur_mod(base_train = base_train,
                                base_test = base_test,
                                y_col = y_col,
                                X_col = X_col,
                                model_name = model_name,
                                model = model,
                                params = params,
                                random_state = random_state,
                                train_size = train_size,
                                test_size = test_size,
                                random_split = random_split,
                                scoring = scoring,
                                cv = cv,
                                n_jobs = n_jobs,
                                refit = refit,
                                verbose = verbose
                                )
        
        print('Outputting classifictions ...')
        
        # create the test classification dataset
        predictions = pd.DataFrame()
        predictions[y_col[0]] = base_test[y_col[0]] .astype(int)
        predictions['PassengerId'] = base_test['PassengerId']
        predictions = predictions[['PassengerId', y_col[0]]]
        
        # output the dataset
        predictions.to_csv(pred_fpath.format(model_name),
                           sep = cons.sep,
                           encoding = cons.encoding,
                           header = cons.header,
                           index = cons.index
                           )
        
    return 0

if __name__ == '__main__':
    
    # extract file paths from constants.py module
    base_engin_fpath = cons.base_engin_data_fpath
    pred_fpath = cons.pred_data_fpath
    
    # extract out model constants
    random_state = cons.random_state
    train_size = cons.train_size
    test_size = cons.test_size
    random_split = cons.random_split
    scoring = 'accuracy'
    refit = cons.refit
    verbose = cons.verbose
    cv = cons.cv
    n_jobs = cons.n_jobs

    # run calssification model
    class_model(base_engin_fpath = base_engin_fpath,
                pred_fpath = pred_fpath,
                random_state = random_state,
                train_size = train_size,
                test_size = test_size,
                random_split = random_split,
                scoring = scoring,
                refit = refit,
                verbose = verbose,
                cv = cv,
                n_jobs = n_jobs
                )
