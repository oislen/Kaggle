# -*- coding: utf-8 -*-
"""
Created on Sun Nov  4 19:18:53 2018

@author: oislen
"""

# load in relevant libraries
from graph.hist import hist
from model.fit_age_mod import fit_age_mod
import cons as cons

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
    
    # plot age distribution before imputing
    hist(dataset = base,
         num_var = ['Age'],
         title = 'Histogram of Age - Pre Imputation'
         )
    
    # split the training data on whether age is missing or not
    base_train = base[base.Age.notnull()]
    base_test = base[base.Age.isnull()]
    
    # extract out model and params
    age_dict = cons.age_dict
    
    # set model constants
    y_col = ['Age']
    X_col =  ['Pclass', 'SibSp', 'Parch', 'FamSize', 'Fare', 'Alone', 'Mr', 'Mrs', 'Ms', 'Priv', 'Male', 'Embarked_Ord']
    model = age_dict['rfr']['model']
    params = age_dict['rfr']['params']
    target_type = 'reg'
    random_state = 123
    train_size = 0.8
    test_size = 0.2
    random_split = True
    sample_target = None
    scoring = 'neg_mean_squared_error'
    refit = True
    verbose = 3
    cv = 10
    n_jobs = -1
    
    # run age na fill model
    base_out = fit_age_mod(base_train = base_train,
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
                           scoring = scoring,
                           refit = refit,
                           cv = cv,
                           n_jobs = n_jobs,
                           verbose = verbose
                           )

    # plot age distribution after imputing
    hist(dataset = base_out,
         num_var = ['Age'],
         title = 'Histogram of Age - Post Imputation'
         )
    
    return base_out
