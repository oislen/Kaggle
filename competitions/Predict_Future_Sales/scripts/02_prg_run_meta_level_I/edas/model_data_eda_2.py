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
top_n_attr = feat_imp['attr'].tolist()

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


dist_comp(attr = 'item_cnt_day_shift_1') # fine
dist_comp(attr = 'year_mean_enc') # useless
dist_comp(attr = 'month_mean_enc') # useless (multiple-values broken?)
dist_comp(attr = 'item_id_months_first_rec') # fine
dist_comp(attr = 'item_cnt_day_shift_1_div_item_id_total_item_cnt_day_shift_1')
dist_comp(attr = 'date_block_num') # useless
dist_comp(attr = 'shop_item_id_mean_enc') # fine
dist_comp(attr = 'year') # useless
dist_comp(attr = 'price_decimal_len') # fine
dist_comp(attr = 'price_decimal') # fine
dist_comp(attr = 'shop_id_item_category_id_total_item_cnt_day_shift_1') # fine
dist_comp(attr = 'item_id_total_item_cnt_day_shift_1') # fine
dist_comp(attr = 'revenue_shift_1') # fine
dist_comp(attr = 'item_cnt_day_shift_1_div_shop_id_total_item_cnt_day_shift_1') # fine
dist_comp(attr = 'city_mean_enc') # fine
dist_comp(attr = 'item_cat_mean_enc') # fine
dist_comp(attr = 'date_block_num_mean_enc') # useless (multiple-values broken?)
dist_comp(attr = 'item_id_mean_enc') # fine
dist_comp(attr = 'item_cat_sub_id') # useless
dist_comp(attr = 'item_category_id_mean_enc') # fine
dist_comp(attr = 'item_cat_sub_mean_enc') # fine
dist_comp(attr = 'city_enc_total_item_cnt_day_shift_1') # fine
dist_comp(attr = 'shop_id') # useless
dist_comp(attr = 'shop_id_mean_enc') # fine
dist_comp(attr = 'item_id') # useless
dist_comp(attr = 'item_id_total_item_cnt_day_shift_3') # fine
dist_comp(attr = 'item_price_shift_1') # fine
dist_comp(attr = 'item_category_id_total_item_cnt_day_shift_1') # fine
dist_comp(attr = 'days_of_month') # useless
dist_comp(attr = 'item_id_city_enc_total_item_cnt_day_shift_1') # fine
dist_comp(attr = 'item_cnt_day_shift_2') # fine
dist_comp(attr = 'delta_item_cnt_day_1_2') # fine
dist_comp(attr = 'item_cnt_day_shift_3_div_shop_id_total_item_cnt_day_shift_3') # fine
dist_comp(attr = 'item_cnt_day_shift_3_div_item_id_total_item_cnt_day_shift_3') # fine
dist_comp(attr = 'item_cnt_day_shift_4_div_shop_id_total_item_cnt_day_shift_4') # fine
dist_comp(attr = 'item_cnt_day_shift_2_div_item_id_total_item_cnt_day_shift_2') # fine
dist_comp(attr = 'item_cnt_day_shift_4_div_item_id_total_item_cnt_day_shift_4') # fine
dist_comp(attr = 'item_cnt_day_shift_2_div_shop_id_total_item_cnt_day_shift_2') # fine
dist_comp(attr = 'item_cat_id') # useless
dist_comp(attr = 'delta_item_cnt_day_3_4') # fine 
dist_comp(attr = 'item_id_months_last_rec') # fine
dist_comp(attr = 'city_enc') # useless (values look brokem)
dist_comp(attr = 'month') # useless
dist_comp(attr = 'n_weekenddays') # fine (possibly remove)
dist_comp(attr = 'n_publicholidays') # fine (possibly remove)
dist_comp(attr = 'totalholidays') # fine (possibly remove)
dist_comp(attr = 'item_category_id') # useless
dist_comp(attr = 'item_cnt_day_shift_3') # fine
dist_comp(attr = 'delta_item_cnt_day_2_3') # fine
dist_comp(attr = 'item_cnt_day_shift_4') # fine
dist_comp(attr = 'shop_id_total_item_cnt_day_shift_1')
dist_comp(attr = 'shop_id_total_item_cnt_day_shift_2')
dist_comp(attr = 'shop_id_total_item_cnt_day_shift_3')
dist_comp(attr = 'shop_id_total_item_cnt_day_shift_4')
dist_comp(attr = 'item_id_total_item_cnt_day_shift_2')
dist_comp(attr = 'item_id_total_item_cnt_day_shift_4')