# -*- coding: utf-8 -*-
"""
Created on Sun Nov  4 22:19:35 2018

@author: oislen
"""

# load in relevant libraries
import pandas as pd
from utilities.fit_mod import fit_mod
import cons
import model_cons

def class_gm(base_engin_fpath,
             pred_fpath
             ):
    
    """
    
    Classification GBM Documentation
    
    Function Overview
    
    This function trains a GBM model classifying the test data
    
    Defaults
    
    class_gm(base_engin_fpath,
             pred_fpath
             )
    
    Parameters
    
    base_engin_fpath - String, the input file path for the base engineered data
    pred_fpath - String, the output file path for the GBM classifications
    
    Returns
    
    0 for successful execution
    
    Example
    
    class_gm(base_engin_fpath = 'C:\\Users\\...\\base_engin.csv',
             pred_fpath = 'C:\\Users\\...\\preds.csv'
             )
        
    """
    
    # load in data
    base = pd.read_csv(base_engin_fpath, 
                       sep = '|'
                       )
    
    # split the data based on the original dataset
    base_train = base[base.Dataset == 'train']
    base_test = base[base.Dataset == 'test']
    
    
    # extract out model and params
    sur_dict = model_cons.sur_dict
    
    # set model constants
    y_col = ['Survived']
    X_col =  ['Pclass', 'SibSp', 'Parch', 'FamSize', 'Fare', 'Alone', 'Mr', 'Mrs', 'Ms', 'Priv', 'Male', 'Embarked']
    model = sur_dict['gbm']['model']
    params = sur_dict['gbm']['params']
    target_type = 'class'
    random_state = 123
    train_size = 0.8
    test_size = 0.2
    random_split = True
    sample_target = 'Survived'
    sample_type = 'over'
    scoring = 'accuracy'
    refit = True
    return_mod = True
    verbose = 3
    cv = 10
    n_jobs = -1
    
    # fit model
    base_out = fit_mod(base_train = base_train,
                       base_test = base_test,
                       y_col = y_col,
                       X_col = X_col,
                       model = model,
                       params = params,
                       target_type = target_type,
                       random_state = random_state,
                       train_size = train_size,
                       test_size = test_size,
                       random_split = random_split,
                       sample_target = sample_target,
                       sample_type = sample_type,
                       scoring = scoring,
                       cv = cv,
                       n_jobs = n_jobs,
                       refit = refit,
                       return_mod = return_mod,
                       verbose = verbose
                       )
    
    # extract out test data
    base_test = base_out.loc[base_out['Dataset'] == 'test', :]
    
    print('Outputting classifictions ...')
    
    # create the test classification dataset
    predictions = pd.DataFrame()
    predictions['Survived'] = base_test['Survived'] .astype(int)
    predictions['PassengerId'] = base_test['PassengerId']
    predictions = predictions[['PassengerId', 'Survived']]
    
    # output the dataset
    predictions.to_csv(pred_fpath,
                       sep = ',',
                       encoding = 'utf-8',
                       header = True,
                       index = False
                       )
    
    return 0

if __name__ == '__main__':
    
    class_gm(base_engin_fpath = cons.base_engin_data_fpath,
             pred_fpath = cons.pred_data_fpath
             )