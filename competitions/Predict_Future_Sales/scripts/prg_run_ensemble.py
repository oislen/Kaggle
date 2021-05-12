# -*- coding: utf-8 -*-
"""
Created on Sun May 17 10:57:04 2020

@author: oislen
"""

# import relevant libraries
import os 
import sys
from importlib import import_module
import cons
import numpy as np
from meta_level_I.exe_model import exe_model
#exe_model = import_module(name = '02_prg_run_ensemble.meta_level_I.exe_model')

np.random.seed(1234)
# user settings
skip_train = False
n = 30
start = 2
end = 2
run_range = range(start, end + 1)
n_cpu = -1

# get current wd
cwd = os.getcwd()

# append required paths
prep_raw_data_path = '{}/02_prg_run_ensemble'.format(cwd)
reference_path = '{}/02_prg_run_ensemble/reference'.format(cwd)
exe_path = '{}/02_prg_run_ensemble/meta_level_I'.format(cwd)
sys.path.append(prep_raw_data_path)
sys.path.append(reference_path)
sys.path.append(exe_path)

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
    # execute the model
    exe_model(cons = cons, feat_imp = 'randforest', max_dept = 3, rand_state = 1, model_type = 'dtree', n = n, skip_train = skip_train, n_cpu = n_cpu, date = date)
    exe_model(cons = cons, feat_imp = 'randforest', max_dept = 5, rand_state = 2, model_type = 'dtree', n = n, skip_train = skip_train, n_cpu = n_cpu, date = date)
    exe_model(cons = cons, feat_imp = 'randforest', max_dept = 7, rand_state = 3, model_type = 'dtree', n = n, skip_train = skip_train, n_cpu = n_cpu, date = date)

#if 3 in run_range:
    #print('~~~~~ Working on KNN Models ...')
    #exe_model(cons = cons, feat_imp = 'randforest', rand_state = 1, model_type = 'knn', n = n, skip_train = skip_train, n_cpu = n_cpu, date = date)
    
if 4 in run_range:
    
    print('~~~~~ Working on Random Forest Models ...')
    
    exe_model(cons = cons, feat_imp = 'randforest', max_dept = 3, rand_state = 1, model_type = 'randforest', n = n, skip_train = skip_train, n_cpu = n_cpu, date = date)
    exe_model(cons = cons, feat_imp = 'randforest', max_dept = 5, rand_state = 2, model_type = 'randforest', n = n, skip_train = skip_train, n_cpu = n_cpu, date = date)
    exe_model(cons = cons, feat_imp = 'randforest', max_dept = 7, rand_state = 3, model_type = 'randforest', n = n, skip_train = skip_train, n_cpu = n_cpu, date = date)
    
if 5 in run_range:
    
    print('~~~~~ Working on Gradient Boosting Models ...')
    
    exe_model(cons = cons, feat_imp = 'gradboost', max_dept = 3, rand_state = 1, model_type = 'gradboost', n = n, skip_train = skip_train, n_cpu = n_cpu, date = date)
    exe_model(cons = cons, feat_imp = 'gradboost', max_dept = 5, rand_state = 2, model_type = 'gradboost', n = n, skip_train = skip_train, n_cpu = n_cpu, date = date)
    exe_model(cons = cons, feat_imp = 'gradboost', max_dept = 7, rand_state = 3, model_type = 'gradboost', n = n, skip_train = skip_train, n_cpu = n_cpu, date = date)
    