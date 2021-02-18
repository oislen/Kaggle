# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 10:42:07 2021

@author: oislen
"""

# load libraries
import os
import sys

# set programme constants
comp_name = 'house-prices-advanced-regression-techniques'
download_data = True
unzip_data = True
del_zip = True

# set directories
git_dir = 'C:\\Users\\User\\Documents\\GitHub'
root_dir = os.path.join(git_dir, 'Kaggle')
utilities_dir = os.path.join(root_dir, 'utilities')
titanic_comp_dir = os.path.join(root_dir, 'competitions\\HousePrices')
scripts_dir = os.path.join(titanic_comp_dir, 'scripts')
data_dir = os.path.join(titanic_comp_dir, 'data')

# append utilities directory to path
for p in [utilities_dir]:
    sys.path.append(p)
