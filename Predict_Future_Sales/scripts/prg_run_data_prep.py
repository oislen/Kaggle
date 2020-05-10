# -*- coding: utf-8 -*-
"""
Created on Sun May 10 18:12:12 2020

@author: oislen
"""

# import relevant libraries
import os 
import sys
from importlib import import_module

# get current wd
cwd = os.getcwd()

# append required paths
prep_raw_data_path = '{}\\01_prg_run_data_prep'.format(cwd)
reference_path = '{}\\01_prg_run_data_prep\\reference'.format(cwd)
sys.path.append(prep_raw_data_path)
sys.path.append(reference_path)

# load in required modules
prg_01 = import_module(name = '01_prep_raw_data')
prg_02 = import_module(name = '02_create_base_data')
prg_03 = import_module(name = '03_agg_base_data')
prg_04 = import_module(name = '04b_backfill_missing_items')
prg_05 = import_module(name = '05_append_supplement_attrs')

# load in file constants
cons = import_module(name = 'file_constants')

# run data prep steps
print('~~~~~ Preparing Raw data ...')
prg_01.prep_raw_data(cons) 
  
print('~~~~~ Creating Base data ...')
prg_02.create_base_data(cons)

print('~~~~~ Aggregating Base data ....')
prg_03.agg_base_data(cons)

print('~~~~~ Back Filling Base data ....')
prg_04.back_fill_missing_items(cons)

print('~~~~~ Append supplemenatary data ...')
prg_05.append_supplement_attrs(cons)

print("~~~~~ Done ...")

    