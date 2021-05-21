# -*- coding: utf-8 -*-
"""
Created on Sun May 17 10:57:04 2020

@author: oislen
"""

# import relevant libraries
import cons
from exe_model import exe_model
from reference.gen_feature_selection import gen_feature_selection

# user True
skip_train = True # whether to skip training and make predictions with pre-fitted models
n = 30 # number of top ranked feature importance attributes to consdier
start = 2 # which step to start from
end = 5 # which step to end at

# generate the run range
run_range = range(start, end + 1)

if 1 in run_range:
    print('~~~~~ Generating feature importance ...')
    # run feature importance
    gen_feature_selection(feat_type = 'randforest') 
    gen_feature_selection(feat_type = 'gradboost') 
    
if 2 in run_range:
    print('~~~~~ Working on Decision Tree Models ...')
    # execute the model
    exe_model(feat_imp = 'randforest', model_type = 'dtree', n = n, skip_train = skip_train)

#if 3 in run_range:
#    print('~~~~~ Working on KNN Models ...')
#    exe_model(feat_imp = 'randforest', model_type = 'knn', n = n, skip_train = skip_train)
    
if 4 in run_range:
    print('~~~~~ Working on Random Forest Models ...')
    exe_model(feat_imp = 'randforest', model_type = 'randforest', n = n, skip_train = skip_train)
    
if 5 in run_range:
    print('~~~~~ Working on Gradient Boosting Models ...')
    exe_model(feat_imp = 'gradboost', model_type = 'gradboost', n = n, skip_train = skip_train)
    