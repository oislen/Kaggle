# -*- coding: utf-8 -*-
"""
Created on Sun May 16 16:29:34 2021

@author: oislen
"""

# load in libraries
import sys
import pandas as pd 
comp_dir = 'C:\\Users\\User\\Documents\\GitHub\\Kaggle\\competitions\\Predict_Future_Sales\\scripts'
sys.path.append(comp_dir)
import cons

# set pandas options
pd.set_option('display.max_rows', 100)

# load in the base model data
base = pd.read_feather(cons.model_data_fpath)
feat_imp = pd.read_csv(cons.randforest_feat_imp)

# extract out top 10 attributes
top_n_attr = feat_imp['attr'][0:10].tolist()

base.columns
base.data_split.value_counts()
base.holdout_subset_ind.value_counts()
base.date_block_num.value_counts()

# create pandas cross table
pd.crosstab(index = base.data_split, columns = base.holdout_subset_ind)
pd.crosstab(index = base.date_block_num, columns = base.holdout_subset_ind)

# extract out the train, validation and holdout sets
train = base.loc[~base.date_block_num.isin([33, 34]), :]
valid = base.loc[base.date_block_num.isin([33]), :]
holdout = base.loc[base.date_block_num.isin([34]), :]

holdout.columns
holdout.isnull().sum().value_counts()

def perc_zero_values(data, sub_cols = None): 
    if sub_cols == None:
        sub_cols = data.columns
    output = (data[sub_cols] == 0).sum() / data.shape[0]
    return output

# check number of zeros
perc_zero_values(data = train, sub_cols = top_n_attr)
perc_zero_values(data = valid, sub_cols = top_n_attr)
perc_zero_values(data = holdout, sub_cols = top_n_attr)

holdout['item_cnt_day_shift_1'].value_counts()


train[top_n_attr]

valid_agg = pd.DataFrame(valid[top_n_attr].max(), columns = ['max'])
holdout_agg = pd.DataFrame(holdout[top_n_attr].max(), columns = ['max'])
data = pd.merge(left = valid_agg, 
                right = holdout_agg, 
                left_index = True, 
                right_index = True
                )

['item_cnt_day_shift_1',
 'year_mean_enc',
 'month_mean_enc',
 'item_id_months_first_rec',
 'item_cnt_day_shift_1_div_item_id_total_item_cnt_day_shift_1',
 'date_block_num',
 'shop_item_id_mean_enc',
 'year',
 'price_decimal_len',
 'price_decimal']

def dist_comp(attr):
    valid_agg = pd.DataFrame(valid[attr].value_counts().sort_index())
    holdout_agg = pd.DataFrame(holdout[attr].value_counts().sort_index())
    data = pd.merge(left = valid_agg, 
                    right = holdout_agg, 
                    left_index = True, 
                    right_index = True,
                    suffixes = ('_valid', '_holdout'),
                    how = 'outer'
                    )
    return data


dist_comp(attr = 'item_cnt_day_shift_1')
dist_comp(attr = 'year_mean_enc')
dist_comp(attr = 'month_mean_enc')
dist_comp(attr = 'item_id_months_first_rec')
