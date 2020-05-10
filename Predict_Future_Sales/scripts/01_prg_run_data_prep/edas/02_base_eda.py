# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 20:52:18 2020

@author: oislen
"""

import pandas as pd
import file_constants as cons
import clean_constants as clean_cons

pd.set_option('display.max_rows', 30)
pd.set_option('display.max_columns', 30)

base_raw = pd.read_feather(cons.base_raw_data_fpath)

base_raw.columns
base_raw.head()
base_raw.dtypes

# want to aggregate to year, month, shop and product level
agg_base = base_raw.groupby(clean_cons.group_cols, as_index = False).agg(clean_cons.agg_dict)

#-- Generate Average Prices Predictions --#

# create pivot table for item sale by date block
sales_table = pd.pivot_table(data = agg_base, 
                             values = ['item_cnt_day'],
                             index = ['shop_id', 'item_id'],
                             columns = ['date_block_num'],
                             fill_value = 0,
                             dropna = False
                             )

# average across the sales table to find prediction for november
avg_item_sales = sales_table.mean(axis = 1)

# convert to dataframe and rename
avg_item_sales_df = avg_item_sales.reset_index().rename(columns = {0:'item_cnt_month'})

# load in test file
test = pd.read_feather(cons.test_clean_fpath)

# join results to test file
test_results = test.merge(avg_item_sales_df, on = ['shop_id', 'item_id'], how = 'left')

# fill in missings
test_results['item_cnt_month'] = test_results['item_cnt_month'].fillna(0)

# drop unneeded columns
test_results = test_results[['ID', 'item_cnt_month']]

# output to csv file
test_results.to_csv(cons.pred_data_dir + '/avg_monthly_preds.csv', index = False)
