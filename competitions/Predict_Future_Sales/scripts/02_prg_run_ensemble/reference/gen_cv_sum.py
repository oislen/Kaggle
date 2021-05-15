# -*- coding: utf-8 -*-
"""
Created on Sat May 15 08:32:17 2021

@author: oislen
"""

import pandas as pd

def gen_cv_sum(clf, 
               cv_sum_fpath
               ):
    
    """
    
    Generate Cross Validation Summary
    
    Function Overview
    
    This functions generates the cross-validation summary for a given classifier and outputs the results to a desired file path.
    
    Defaults
    
    gen_cv_sum(clf, 
               cv_sum_fpath
               )
    
    Parameters
    
    clf - Sklearn GridSearchCV, the fitted scikit-learn grid search cv object to extract results from
    cv_sum_fpath - String, the output file path to write the cross-validation results to as a .csv file
    
    Returns
    
    cv_results - DataFrame, the cross-validation results
    
    Example
    
    gen_cv_sum(clf = gcv, 
               cv_sum_fpath = cv_sum_fpath
               )
    
    """
    
    # create a DataFrame with relevent results extracted from the given GridSearchCV object
    cv_results = pd.DataFrame({'params':clf.cv_results_['params'],
                               'mean_test_score':clf.cv_results_['mean_test_score'] * -1,
                               'std_test_score':clf.cv_results_['std_test_score'],
                               'rank_test_score':clf.cv_results_['rank_test_score']
                               })
    
    # sort the results by rank test score
    cv_results = cv_results.sort_values(by = ['rank_test_score'])
    
    # write the results to the file path
    cv_results.to_csv(cv_sum_fpath, index = False)
    
    return cv_results
    