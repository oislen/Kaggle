# -*- coding: utf-8 -*-
"""
Created on Sun May 17 10:57:04 2020

@author: oislen
"""

# import relevant libraries
import os 
import sys
from importlib import import_module
import numpy as np

np.random.seed(1234)
# user settings
skip_train = False
n = 30
start = 2
end = 2
run_range = range(start, end + 1)
n_cpu = 6

# get current wd
cwd = os.getcwd()

# append required paths
prep_raw_data_path = '{}/02_prg_run_ensemble'.format(cwd)
reference_path = '{}/02_prg_run_ensemble/reference'.format(cwd)
exe_path = '{}/02_prg_run_ensemble/meta_level_I'.format(cwd)
sys.path.append(prep_raw_data_path)
sys.path.append(reference_path)
sys.path.append(exe_path)

# load in file constants
cons = import_module(name = 'file_constants')

# get todays date
#date = dt.datetime.today().strftime('%Y%m%d')
date = '20200523'

if 1 in run_range:
    
    print('~~~~~ Generating feature importance ...')
    
    prg_feat_imp = import_module(name = 'feature_importance')
    prg_feat_imp.gen_feature_selection(cons,  feat_type = 'randforest', n_cpu = n_cpu) 
    prg_feat_imp.gen_feature_selection(cons,  feat_type = 'gradboost', n_cpu = n_cpu) 
    
if 2 in run_range:
    
    print('~~~~~ Working on Decision Tree Models ...')
    
    prg_dt = import_module(name = 'mod_dtree')
    prg_dt.mod_dtree(cons, max_dept = 3, rand_state = 1, model_type = 'dtree', feat_imp = 'randforest', n = n, date = date, skip_train = skip_train, n_cpu = n_cpu)
    prg_dt.mod_dtree(cons, max_dept = 5, rand_state = 2, model_type = 'dtree', feat_imp = 'randforest', n = n, date = date, skip_train = skip_train, n_cpu = n_cpu)
    prg_dt.mod_dtree(cons, max_dept = 7, rand_state = 3, model_type = 'dtree', feat_imp = 'randforest', n = n, date = date, skip_train = skip_train, n_cpu = n_cpu)
 
#if 3 in run_range:
    
    #print('~~~~~ Working on KNN Models ...')
    
    #prg_knn = import_module(name = 'mod_knn')
    #prg_knn.mod_knn(cons, model_type = 'knn', feat_imp = 'randforest', n = n, date = date, skip_train = skip_train, n_cpu = n_cpu)

if 4 in run_range:
    
    print('~~~~~ Working on Random Forest Models ...')
    
    prg_rf = import_module(name = 'mod_randforest')
    prg_rf.mod_randforest(cons, max_dept = 3, rand_state = 1, model_type = 'randforest', feat_imp = 'randforest', n = n, date = date, skip_train = skip_train, n_cpu = n_cpu) 
    prg_rf.mod_randforest(cons, max_dept = 5, rand_state = 2, model_type = 'randforest', feat_imp = 'randforest', n = n, date = date, skip_train = skip_train, n_cpu = n_cpu) 
    prg_rf.mod_randforest(cons, max_dept = 7, rand_state = 3, model_type = 'randforest', feat_imp = 'randforest', n = n, date = date, skip_train = skip_train, n_cpu = n_cpu) 
    
if 5 in run_range:
    
    print('~~~~~ Working on Gradient Boosting Models ...')
    
    prg_gb = import_module(name = 'mod_gradboost')
    prg_gb.mod_gradboost(cons, max_dept = 3, rand_state = 1, model_type = 'gradboost', feat_imp = 'gradboost', n = n, date = date, skip_train = skip_train, n_cpu = n_cpu)
    prg_gb.mod_gradboost(cons, max_dept = 5, rand_state = 2, model_type = 'gradboost', feat_imp = 'gradboost', n = n, date = date, skip_train = skip_train, n_cpu = n_cpu)
    prg_gb.mod_gradboost(cons, max_dept = 7, rand_state = 3, model_type = 'gradboost', feat_imp = 'gradboost', n = n, date = date, skip_train = skip_train, n_cpu = n_cpu)