# -*- coding: utf-8 -*-
"""
Created on Thu May 21 19:31:18 2020

@author: oislen
"""

import pandas as pd
import reference.utilities_ensemble as utl_ens
from xgboost import XGBRegressor
from xgboost import plot_importance
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
import numpy as np

pd.set_option('display.max_columns', 40)

def plot_features(booster, figsize):    
    fig, ax = plt.subplots(1,1,figsize=figsize)
    return plot_importance(booster=booster, ax=ax)

# set the input data file path
data_fpath = 'C:/Users/User/Documents/GitHub/Kaggle/Predict_Future_Sales/data/model/model_data.feather'

# load in model data
base = pd.read_feather(data_fpath)

# set the train, valid and test sub limits
test_split_dict = {'train_sub':31, 'valid_sub':32, 'test_sub':33}

base.columns

index_cols = ['date_block_num', 'shop_id', 'item_id', 'ID', 'data_split', 
              'primary_key', 'holdout_subset_ind', 'no_sales_hist_ind', 
              'no_holdout_sales_hist_ind'
              ] 

tar_cols = ['item_cnt_day']

pred_cols = ['item_cnt_day_shift_1_div_item_id_total_item_cnt_day_shift_1',
             'price_decimal_len',
             'item_id_city_enc_total_item_cnt_day_shift_1',
             'item_id_total_item_cnt_day_shift_1',
             'item_price_shift_1',
             'item_id_months_first_rec',
             'revenue_shift_1',
             'shop_id',
             'date_block_num',
             'item_id',
             'shop_id_total_item_cnt_day_shift_12',
             'item_cat_sub_enc',
             'delta_item_cnt_day_1_2',
             'month',
             'item_cnt_day_shift_1',
             'item_cnt_day_shift_3_div_item_id_total_item_cnt_day_shift_3',
             'item_category_id_total_item_cnt_day_shift_1',
             'item_cnt_day_shift_2',
             'totalholidays',
             'item_category_id',
             'item_cnt_day_shift_2_div_item_id_total_item_cnt_day_shift_2',
             'year',
             'price_decimal',
             'item_cnt_day_shift_3',
             'shop_id_total_item_cnt_day_shift_2',
             'item_cnt_day_shift_12',
             'item_cnt_day_shift_6',
             'item_id_total_item_cnt_day_shift_2',
             'item_cat_enc',
             'item_cnt_day_shift_4',
             'shop_id_item_category_id_total_item_cnt_day_shift_1',
             'shop_id_total_item_cnt_day_shift_4'
             ]
    
# run the data splits function
data_splits_dict = utl_ens.extract_data_splits(dataset = base,
                                               req_cols = [],
                                               index_cols = index_cols,
                                               tar_cols = tar_cols,
                                               pred_cols = pred_cols,
                                               test_split_dict = test_split_dict
                                               )

# extract out the data splits
X_train = data_splits_dict['X_train']
y_train = data_splits_dict['y_train']
X_valid = data_splits_dict['X_valid']
y_valid = data_splits_dict['y_valid']
X_test = data_splits_dict['X_test']
y_test = data_splits_dict['y_test']
X_holdout = data_splits_dict['X_holdout']
y_holdout = data_splits_dict['y_holdout']

x_preds = ['item_cnt_day_shift_1_div_item_id_total_item_cnt_day_shift_1',
           'price_decimal_len',
           'item_id_city_enc_total_item_cnt_day_shift_1',
           'item_id_total_item_cnt_day_shift_1',
           'item_price_shift_1',
           'item_id_months_first_rec',
           'revenue_shift_1',
           'shop_id',
           'date_block_num',
           'item_id',
           'shop_id_total_item_cnt_day_shift_12',
           'item_cat_sub_enc',
           'delta_item_cnt_day_1_2',
           'month',
           'item_cnt_day_shift_1',
           'item_cnt_day_shift_3_div_item_id_total_item_cnt_day_shift_3',
           'item_category_id_total_item_cnt_day_shift_1',
           'item_cnt_day_shift_2',
           'totalholidays',
           'item_category_id',
           'item_cnt_day_shift_2_div_item_id_total_item_cnt_day_shift_2',
           'year',
           'price_decimal',
           'item_cnt_day_shift_3',
           'shop_id_total_item_cnt_day_shift_2',
           'item_cnt_day_shift_12',
           'item_cnt_day_shift_6',
           'item_id_total_item_cnt_day_shift_2',
           'item_cat_enc',
           'item_cnt_day_shift_4',
           'shop_id_item_category_id_total_item_cnt_day_shift_1',
           'shop_id_total_item_cnt_day_shift_4'
           ]

y_tar = 'item_cnt_day'


model = XGBRegressor(max_depth = 8,
                     n_estimators = 1000,
                     min_child_weight = 300, 
                     colsample_bytree = 0.8, 
                     subsample = 0.8, 
                     eta = 0.3,    
                     seed = 42
                     )

model.fit(X_train[x_preds], 
          y_train[y_tar], 
          eval_metric = "rmse", 
          eval_set = [(X_train[x_preds], y_train[y_tar]), (X_valid[x_preds], y_valid[y_tar])], 
          verbose = True, 
          early_stopping_rounds = 10
          )

y_valid['preds'] = model.predict(X_valid[x_preds]).clip(0, 20)
y_test['preds'] = model.predict(X_test[x_preds]).clip(0, 20)
y_holdout['preds'] = model.predict(X_holdout[x_preds]).clip(0, 20)


# prediction value counts
print('Validation Predictions:')
print(y_valid['preds'].value_counts())
print('Test Predictions:')
print(y_test['preds'].value_counts())
print('Holdout Predictions:')
print(y_holdout['preds'].value_counts())

#-- RMSE --#

# calculate RMSE
valid_rmse = np.sqrt(((y_valid['item_cnt_day'] - y_valid['preds']) ** 2).sum() / y_valid.shape[0])
print('Validation Set RMSE:', valid_rmse)

# calculate RMSE
test_rmse = np.sqrt(((y_test['item_cnt_day'] - y_test['preds']) ** 2).sum() / y_test.shape[0])
print('Test Set RMSE:', test_rmse)

#-- Preds vs True --#

# create confusion matrix
sns.scatterplot(x = 'item_cnt_day', y = 'preds', data = y_valid)
plt.show() 

# create confusion matrix
sns.scatterplot(x = 'item_cnt_day', y = 'preds', data = y_test)
plt.show() 

#-- Pred Hist --#

# create a hist of pred distribution
sns.distplot(a = y_valid['preds'], bins = 100, kde = False)
plt.show() 

sns.distplot(a = y_test['preds'], bins = 100, kde = False)
plt.show() 
    
# create a hist of pred distribution
sns.distplot(a = y_holdout['preds'], bins = 100, kde = False)
plt.show() 

submission = pd.DataFrame({"ID": y_holdout['ID'].astype(int), "item_cnt_month": y_holdout['preds'].rename({'y_holdout_pred':'item_cnt_month'})})
submission = submission.sort_values(by = ['ID'])

submission.to_csv('C:/Users/User/Documents/GitHub/Kaggle/Predict_Future_Sales/data/pred/xgb20200521.csv', index=False)

# save predictions for an ensemble
#pickle.dump(y_pred, open('xgb_train.pickle', 'wb'))
#pickle.dump(y_test, open('xgb_test.pickle', 'wb'))
xgboost_mod = 'C:/Users/User/Documents/GitHub/Kaggle/Predict_Future_Sales/models/xgboost_dept8_model.pkl'
pickle.dump(model, open(xgboost_mod, "wb"))


plot_features(model, (10,14))
