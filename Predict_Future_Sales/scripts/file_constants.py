# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 17:15:06 2020

@author: oislen
"""

# set project directories
git_dir = 'C:/Users/User/Documents/GitHub'
comp_dir = '{}/Kaggle/Predict_Future_Sales'.format(git_dir)
data_dir = '{}/data'.format(comp_dir)
scripts_dir = '{}/scripts'.format(comp_dir)
reports_dir = '{}/report'.format(comp_dir)

# set custom function location
va_dir = '{}/value_analysis'.format(git_dir)

# set data directories
raw_data_dir = '{}/raw'.format(data_dir)
clean_data_dir = '{}/clean'.format(data_dir)
base_data_dir = '{}/base'.format(data_dir)
pred_data_dir = '{}/pred'.format(data_dir)

# set raw data file paths
item_categories_fpath = '{}/item_categories.csv'.format(raw_data_dir)
items_fpath = '{}/items.csv'.format(raw_data_dir)
sales_train_fpath = '{}/sales_train.csv'.format(raw_data_dir)
sample_submission_fpath = '{}/sample_submission.csv'.format(raw_data_dir)
shops_fpath = '{}/shops.csv'.format(raw_data_dir)
test_fpath = '{}/test.csv'.format(raw_data_dir)

# set clean file paths
item_categories_clean_fpath = '{}/item_categories_clean.feather'.format(clean_data_dir)
items_clean_fpath = '{}/items_clean.feather'.format(clean_data_dir)
sales_train_clean_fpath = '{}/sales_train_clean.feather'.format(clean_data_dir)
sample_submission_clean_fpath = '{}/sample_submission_clean.feather'.format(clean_data_dir)
shops_clean_fpath = '{}/shops_clean.feather'.format(clean_data_dir)
test_clean_fpath = '{}/test_clean.feather'.format(clean_data_dir)

# set base file path
base_raw_data_fpath = '{}/base_raw.feather'.format(base_data_dir)
base_agg_data_fpath = '{}/base_agg.feather'.format(base_data_dir)
