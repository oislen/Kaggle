# -*- coding: utf-8 -*-
"""
Created on Sun May 10 19:15:38 2020

@author: oislen
"""

import pandas as pd
import seaborn as sns
from importlib import import_module

pd.set_option('display.max_columns', 30)

# load in file constants
cons = import_module(name = 'file_constants')

# read in the base data
feather_file = 'C:/Users/User/Documents/GitHub/Kaggle/Predict_Future_Sales/data/base/base_supp_data.feather'
# feather_file = cons.base_agg_supp_fpath

base = pd.read_feather(feather_file)

base.columns

base['item_price'].value_counts(dropna = False)
base['item_cnt_day'].value_counts(dropna = False)
base['n_refund'].value_counts(dropna = False)
base['n_sale'].value_counts(dropna = False)
base['data_split'].value_counts(dropna = False)
base['ID'].value_counts(dropna = False)
base['holdout_subset_ind'].value_counts(dropna = False)

pd.crosstab(index = base['no_sales_hist_ind'], columns = base['item_cnt_day'].isin([0, -999]).astype(int))
pd.crosstab(index = base['no_sales_hist_ind'], columns = base['holdout_subset_ind'])

# Step 1:
# TODO: extract out below section into a seperate script
# TODO: encode categorical variables; alphabetical / order encode
# TODO: create a change in price attribute
# TODO: create shift attribues; last month, last three months, last year
# TODO: create shop / item sales totals
# TODO: create date window attributes; quarters, seasons
# TODO: create mean encoded attributes; item category, item_id, shop_id
# TODO: create interaction attributes

from sklearn import preprocessing

# label encode  item cat
item_cat_label_enc = preprocessing.LabelEncoder()
item_cat_label_enc.fit(base['item_cat'].unique())
base['item_cat_enc'] = item_cat_label_enc.transform(base['item_cat'])

# label encode item cat sub
item_cat_sub_label_enc = preprocessing.LabelEncoder()
item_cat_sub_label_enc.fit(base['item_cat_sub'].unique())
base['item_cat_sub_enc'] = item_cat_sub_label_enc.transform(base['item_cat_sub'])

# Step 2:
# TODO: extract out below section into a seperate script
# TODO: split into train, valid, test and holdout
# TODO: fit random forest and gradient boosting decision trees; loss optimise on MSE
# TODO: plot early stopping plot
# TODO: validate with prediction vs true plot
# TODO: validate with RMSE metric
# TODO: tune hyperparameters
# TODO: apply stacking with linear meta-model
# TODO: predict any item with no sales history as 0

# define data split filters
filt_train = base['data_split'] == 'train'
filt_valid = base['data_split'] == 'valid'
filt_test = base['data_split'] == 'test'
filt_holdout = base['data_split'] == 'holdout'

# extract out the data splits
train_data = base[filt_train]
valid_data = base[filt_valid]
test_data = base[filt_test]
holdout_data = base[filt_holdout]

# seperate predictors from response
data_cols = base.columns.tolist()
index_cols = ['primary_key', 'ID', 'data_split', 'holdout_subset_ind', 'no_sales_hist_ind']
tar_cols = ['item_cnt_day', 'n_refund', 'n_sale']
excl_cols = ['item_category_id', 'item_cat', 'item_cat_sub']
pred_cols = [col for col in data_cols if col not in index_cols + tar_cols + excl_cols]

# split datasets into train, valid, test and holdout
X_train = train_data[index_cols + pred_cols]
y_train = train_data[index_cols + tar_cols]
X_valid = valid_data[index_cols + pred_cols]
y_valid = valid_data[index_cols + tar_cols]
X_test = test_data[index_cols + pred_cols]
y_test = test_data[index_cols + tar_cols]
X_holdout = holdout_data[index_cols + pred_cols]
y_holdout = holdout_data[index_cols + tar_cols]

from sklearn.ensemble import RandomForestRegressor

# initiate random forest model
rfc = RandomForestRegressor(max_depth = 7, 
                            random_state = 1234, 
                            criterion = 'mse',
                            n_estimators = 100,
                            n_jobs = 2,
                            verbose = 2,
                            max_features = 'auto'
                            )

# fit random forests model
rfc.fit(X_train[pred_cols], y_train['item_cnt_day'])

# make predictions for valid, test and holdout
y_valid['y_valid_pred'] = rfc.predict(X_valid[pred_cols])
#y_test['y_test_pred'] = rfc.predict(X_test[pred_cols])
y_holdout['y_holdout_pred'] = rfc.predict(X_holdout[pred_cols])


# TODO: incorporate a whole script in model evaluation here
y_holdout['y_holdout_pred'].value_counts()

# create confusion matrix
pd.crosstab(index = y_valid['item_cnt_day'], 
            columns = y_valid['y_valid_pred']
            )

# create confusion matrix
sns.scatterplot(x = 'item_cnt_day', y = 'y_valid_pred', data = y_valid)


# TODO: possibly extract this last step out into a seperate script
# map items with no historical sell to 0
y_holdout.columns
no_sales_hist_filt = y_holdout['no_sales_hist_ind'] == 1
y_holdout.loc[no_sales_hist_filt, ['y_holdout_pred']] = 0

# extract out test predictions
holdout_subset_filt = y_holdout['holdout_subset_ind'] == 1
holdout_out = y_holdout.loc[holdout_subset_filt, ['ID', 'y_holdout_pred']]
holdout_out = holdout_out.rename(columns = {'y_holdout_pred':'item_cnt_month'})
holdout_out_sort = holdout_out.sort_values(by = ['ID']).astype(int)

# output predictions as csv file
output_foath = cons.pred_data_dir + '/randforest20200514.csv'
holdout_out_sort.to_csv(output_foath,
                        index = False
                        )
