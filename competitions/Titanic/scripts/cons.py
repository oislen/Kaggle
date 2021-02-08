# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 17:20:45 2021

@author: oislen
"""

# load libraries
import os
import sys

# set programme constants
comp_name = 'titanic'
download_data = True
unzip_data = True
del_zip = True

# set directories
git_dir = 'C:\\Users\\User\\Documents\\GitHub'
root_dir = os.path.join(git_dir, 'Kaggle')
va_dir = os.path.join(git_dir, 'value_analysis')
utilities_dir = os.path.join(root_dir, 'utilities')
titanic_comp_dir = os.path.join(root_dir, 'competitions\\Titanic')
scripts_dir = os.path.join(titanic_comp_dir, 'scripts')
data_dir = os.path.join(titanic_comp_dir, 'data')
report_dir = os.path.join(titanic_comp_dir, 'report')
univar_stats_dir = os.path.join(report_dir, 'univariate_analysis\\descriptive_stats')
bivar_assoc_dir = os.path.join(report_dir, 'bivariate_analysis\\association_tests')
bivar_corr_dir = os.path.join(report_dir, 'bivariate_analysis\\correlation_tests')
bivar_gains_dir = os.path.join(report_dir, 'bivariate_analysis\\measure_gains')

# define filenames
zip_data_fname = '{}.zip'.format(comp_name)
sample_sub_data_fname = 'gender_submission.csv'
test_data_fname = 'test.csv'
train_data_fname = 'train.csv'
base_data_fname = 'base.csv'
base_clean_data_fname = 'base_clean.csv'
base_clean_2_data_fname = 'base_clean_2.csv'
base_engin_fname = 'base_engin.csv'
pred_data_fname = 'preds.csv'

# create file paths
zip_data_fpath = os.path.join(data_dir, zip_data_fname)
sample_sub_data_fpath = os.path.join(data_dir, sample_sub_data_fname)
test_data_fpath = os.path.join(data_dir, test_data_fname)
train_data_fpath = os.path.join(data_dir, train_data_fname)
base_data_fpath = os.path.join(data_dir, base_data_fname)
base_clean_data_fpath = os.path.join(data_dir, base_clean_data_fname)
base_clean_2_data_fpath = os.path.join(data_dir, base_clean_2_data_fname)
base_engin_data_fpath = os.path.join(data_dir, base_engin_fname)
pred_data_fpath = os.path.join(data_dir, pred_data_fname)

# append utilities directory to path
for p in [utilities_dir, va_dir]:
    sys.path.append(p)