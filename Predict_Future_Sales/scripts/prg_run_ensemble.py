# -*- coding: utf-8 -*-
"""
Created on Sun May 17 10:57:04 2020

@author: oislen
"""

# import relevant libraries
import os 
import sys
from importlib import import_module

# get current wd
cwd = os.getcwd()

# append required paths
prep_raw_data_path = '{}\\02_prg_run_ensemble'.format(cwd)
reference_path = '{}\\02_prg_run_ensemble\\reference'.format(cwd)
sys.path.append(prep_raw_data_path)
sys.path.append(reference_path)

# load in required modules
prg_04 = import_module(name = '04_model_validation')
prg_05 = import_module(name = '05_format_kaggle_preds')

# load in file constants
cons = import_module(name = 'file_constants')
