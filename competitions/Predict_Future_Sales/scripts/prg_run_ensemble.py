# -*- coding: utf-8 -*-
"""
Created on Sun May 17 10:57:04 2020

@author: oislen
"""

# import relevant libraries
import cons
from exe_model import exe_model
from reference.gen_feature_selection import gen_feature_selection

# user settings
skip_train = False # whether to skip training and make predictions with pre-fitted models
n = 30 # number of top ranked feature importance attributes to consdier
start = 2 # which step to start from
end = 2 # which step to end at
date = '20200523' # set the date to output the files with
#date = dt.datetime.today().strftime('%Y%m%d') # alternatively use todays date

# generate the run range
run_range = range(start, end + 1)

if 1 in run_range:
    
    print('~~~~~ Generating feature importance ...')
    
    # run feature importance
    gen_feature_selection(cons,  feat_type = 'randforest', n_cpu = cons.n_cpu) 
    gen_feature_selection(cons,  feat_type = 'gradboost', n_cpu = cons.n_cpu) 
    
if 2 in run_range:
    print('~~~~~ Working on Decision Tree Models ...')
    # execute the model
    exe_model(cons = cons, feat_imp = 'randforest', max_dept = 3, rand_state = 1, model_type = 'dtree', n = n, skip_train = skip_train, n_cpu = cons.n_cpu, date = date)
    exe_model(cons = cons, feat_imp = 'randforest', max_dept = 5, rand_state = 2, model_type = 'dtree', n = n, skip_train = skip_train, n_cpu = cons.n_cpu, date = date)
    exe_model(cons = cons, feat_imp = 'randforest', max_dept = 7, rand_state = 3, model_type = 'dtree', n = n, skip_train = skip_train, n_cpu = cons.n_cpu, date = date)

if 3 in run_range:
    print('~~~~~ Working on KNN Models ...')
    exe_model(cons = cons, feat_imp = 'randforest', rand_state = 1, model_type = 'knn', n = n, skip_train = skip_train, n_cpu = cons.n_cpu, date = date)
    
if 4 in run_range:
    
    print('~~~~~ Working on Random Forest Models ...')
    
    exe_model(cons = cons, feat_imp = 'randforest', max_dept = 3, rand_state = 1, model_type = 'randforest', n = n, skip_train = skip_train, n_cpu = cons.n_cpu, date = date)
    exe_model(cons = cons, feat_imp = 'randforest', max_dept = 5, rand_state = 2, model_type = 'randforest', n = n, skip_train = skip_train, n_cpu = cons.n_cpu, date = date)
    exe_model(cons = cons, feat_imp = 'randforest', max_dept = 7, rand_state = 3, model_type = 'randforest', n = n, skip_train = skip_train, n_cpu = cons.n_cpu, date = date)
    
if 5 in run_range:
    
    print('~~~~~ Working on Gradient Boosting Models ...')
    
    exe_model(cons = cons, feat_imp = 'gradboost', max_dept = 3, rand_state = 1, model_type = 'gradboost', n = n, skip_train = skip_train, n_cpu = cons.n_cpu, date = date)
    exe_model(cons = cons, feat_imp = 'gradboost', max_dept = 5, rand_state = 2, model_type = 'gradboost', n = n, skip_train = skip_train, n_cpu = cons.n_cpu, date = date)
    exe_model(cons = cons, feat_imp = 'gradboost', max_dept = 7, rand_state = 3, model_type = 'gradboost', n = n, skip_train = skip_train, n_cpu = cons.n_cpu, date = date)
    