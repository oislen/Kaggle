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

# set .csv constants
sep = ','
encoding = 'utf-8'
header = True
index = False

# set directories
git_dir = 'C:\\Users\\User\\Documents\\GitHub'
root_dir = os.path.join(git_dir, 'Kaggle')
utilities_dir = os.path.join(root_dir, 'utilities')
houseprices_comp_dir = os.path.join(root_dir, 'competitions\\HousePrices')
scripts_dir = os.path.join(houseprices_comp_dir, 'scripts')
data_dir = os.path.join(houseprices_comp_dir, 'data')

# create file names
train_data_fname = 'train.csv'
test_data_fname = 'test.csv'
base_data_fname = 'base.csv'

# create file paths
train_data_fpath = os.path.join(data_dir, train_data_fname)
test_data_fpath = os.path.join(data_dir, test_data_fname)
base_data_fpath = os.path.join(data_dir, base_data_fname)

# append utilities directory to path
for p in [utilities_dir]:
    sys.path.append(p)
