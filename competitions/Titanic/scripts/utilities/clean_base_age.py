# -*- coding: utf-8 -*-
"""
Created on Sun Nov  4 19:18:53 2018

@author: oislen
"""

# load in relevant libraries
import cons
from utilities.age_nafill_mod import age_nafill_mod

def clean_base_age(base):
        
    """

    Clean Base Age Documentation

    Function Overview
    
    This function cleans and processes the missing age values.
    
    Defaults
    
    clean_base_age(base)
    
    Parameters
    
    base - DataFrame, the base data to fill age column
    
    Returns
    
    base - DataFrame, the base data with filled age column
    
    Example

    clean_base_age(base = base)

    """
    
    # create null age indicator
    base['age_null_ind'] = base['Age'].isnull()
    
    # check data types
    base.dtypes
    
    # split the data based on the original dataset
    base_train = base[base.Dataset == 'train']
    base_test = base[base.Dataset == 'test']

    # split the training data on whether age is missing or not
    train = base_train[base_train.Age.notnull()]
    test = base_train[base_train.Age.isnull()]
    
    # set model constants
    y_col = ['Age']
    X_col =  ['Survived', 'Pclass', 'SibSp', 'Parch', 'FamSize', 'Fare', 'Alone', 'Mr', 'Mrs', 'Ms', 'Priv', 'male', 'Embarked']
    params = cons.train_age_gbm_params
    random_state = 123
    train_size = 0.8
    test_size = 0.2
    random_split = True
    sample_target = None
    scoring = 'neg_mean_squared_error'
    
    # run age na fill model
    base_train = age_nafill_mod(base_train = train,
                                base_test = test,
                                y_col = y_col,
                                X_col = X_col,
                                params = params,
                                random_state = random_state,
                                train_size = train_size,
                                test_size = test_size,
                                random_split = random_split,
                                sample_target = sample_target,
                                scoring = scoring
                                )
    
    # set model constants
    X_col =  ['Pclass', 'SibSp', 'Parch', 'FamSize', 'Fare', 'Alone', 'Mr', 'Mrs', 'Ms', 'Priv', 'male', 'Embarked']
    params = cons.test_age_gbm_params
    
    # run age na fill model
    base = age_nafill_mod(base_train = base_train,
                          base_test = base_test,
                          y_col = y_col,
                          X_col = X_col,
                          params = params,
                          random_state = random_state,
                          train_size = train_size,
                          test_size = test_size,
                          random_split = random_split,
                          sample_target = sample_target,
                          scoring = scoring
                          )

    # drop missing age indicator
    base = base.drop(columns = ['age_null_ind'])
    
    return base
