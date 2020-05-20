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
start = 1
end = 1
run_range = range(start, end + 1)

# get current wd
cwd = os.getcwd()

# append required paths
prep_raw_data_path = '{}\\02_prg_run_ensemble'.format(cwd)
reference_path = '{}\\02_prg_run_ensemble\\reference'.format(cwd)
exe_path = '{}\\02_prg_run_ensemble\\exe_model'.format(cwd)
sys.path.append(prep_raw_data_path)
sys.path.append(reference_path)
sys.path.append(exe_path)

# load in required modules
prg_01 = import_module(name = '01_feature_selection')
prg_rf = import_module(name = 'mod_randforest')
prg_dt = import_module(name = 'mod_dtree')

# load in file constants
cons = import_module(name = 'file_constants')

if 1 in run_range:
    print('~~~~~ Generating feature importance ...')
    prg_01.gen_feature_selection(cons) 
    
if 2 in run_range:
    print('~~~~~ Fitting, training and predicting ...')
    prg_rf.mod_randforest(cons) 
    
if 3 in run_range:
    print('~~~~~ Fitting, training and predicting ...')
    prg_dt.mod_dtree(cons)