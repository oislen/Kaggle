# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 17:15:06 2020

@author: oislen
"""

import datetime as dt

# set project directories
git_dir = 'C:/Users/User/Documents/GitHub'
comp_dir = '{}/Kaggle/Predict_Future_Sales'.format(git_dir)
data_dir = '{}/data'.format(comp_dir)
scripts_dir = '{}/scripts'.format(comp_dir)
reports_dir = '{}/report'.format(comp_dir)
models_dir = '{}/models'.format(comp_dir)

# set custom function location
va_dir = '{}/value_analysis'.format(git_dir)

# set data directories
raw_data_dir = '{}/raw'.format(data_dir)
clean_data_dir = '{}/clean'.format(data_dir)
base_data_dir = '{}/base'.format(data_dir)
model_data_dir = '{}/model'.format(data_dir)
pred_data_dir = '{}/pred'.format(data_dir)

# set report sub directories
feat_imp_dir = '{}/feat_imp'.format(reports_dir)

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
base_raw_data_fpath = '{}/base_raw_data.feather'.format(base_data_dir)
base_raw_test_fpath = '{}/base_raw_test.feather'.format(base_data_dir)
base_agg_data_fpath = '{}/base_agg_data.feather'.format(base_data_dir)
base_agg_comp_fpath = '{}/base_comp_data.feather'.format(base_data_dir)
base_agg_totl_fpath = '{}/base_totl_data.feather'.format(base_data_dir)
base_agg_shft_fpath = '{}/base_shft_data.feather'.format(base_data_dir)
base_agg_supp_fpath = '{}/base_supp_data.feather'.format(base_data_dir)

# set model data file path
model_data_fpath = '{}/model_data.feather'.format(model_data_dir)

# set feature importance paths
randforest_feat_imp = '{}/randforest_feat_imp.csv'.format(feat_imp_dir)

# set predictions
date = dt.datetime.today().strftime('%Y%m%d')
randforest_preds = '{}/randforest{}'.format(pred_data_dir, date)
dtree_preds = '{}/dtree{}'.format(pred_data_dir, date)
