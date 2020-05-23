# -*- coding: utf-8 -*-
"""
Created on Sun May 17 10:57:04 2020

@author: oislen
"""

# import relevant libraries
import os 
import sys
from importlib import import_module

# user settings
start = 2
end = 2
run_range = range(start, end + 1)

# get current wd
cwd = os.getcwd()

# append required paths
prep_raw_data_path = '{}\\02_prg_run_ensemble'.format(cwd)
reference_path = '{}\\02_prg_run_ensemble\\reference'.format(cwd)
exe_path = '{}\\02_prg_run_ensemble\\meta_level_I'.format(cwd)
sys.path.append(prep_raw_data_path)
sys.path.append(reference_path)
sys.path.append(exe_path)

# load in required modules
prg_01 = import_module(name = 'feature_importance')

prg_rf3 = import_module(name = 'mod_randforest3')
prg_rf5 = import_module(name = 'mod_randforest5')
prg_rf7 = import_module(name = 'mod_randforest7')

prg_dt3 = import_module(name = 'mod_dtree3')
prg_dt5 = import_module(name = 'mod_dtree5')
prg_dt7 = import_module(name = 'mod_dtree7')

prg_gb = import_module(name = 'mod_gradboost')

# load in file constants
cons = import_module(name = 'file_constants')

if 1 in run_range:
    print('~~~~~ Generating feature importance ...')
    prg_01.gen_feature_selection(cons) 
    
if 2 in run_range:
    print('~~~~~ Working on Random Forest Models ...')
    prg_rf3.mod_randforest3(cons) 
    prg_rf5.mod_randforest5(cons) 
    prg_rf7.mod_randforest7(cons) 
    
if 3 in run_range:
    print('~~~~~ Working on Decision Tree Models ...')
    prg_dt3.mod_dtree3(cons)
    prg_dt5.mod_dtree5(cons)
    prg_dt7.mod_dtree7(cons)
    
if 4 in run_range:
    print('~~~~~ Fitting, training and predicting ...')
    prg_gb.mod_gradboost(cons)