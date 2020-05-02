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

#-- Create Lagged Attributes --#

def gen_shift_attr(data):
    
    data = agg_base.copy(True)
    
    # create pivot table for item sale by date block
    sales_table = pd.pivot_table(data = data, 
                                 values = ['item_cnt_day'],
                                 index = ['date_block_num'],
                                 columns = ['shop_id', 'item_id'],
                                 fill_value = 0,
                                 dropna = False
                                 )
    
    for i in sales_table.index:
        print('working on {} ...'.format(i))
        # remove the columns all mising 
        cols = sales_table.columns
        non_zero_cols = ~(sales_table == 0).all()
        sales_table_filt = sales_table[cols[non_zero_cols]]
        shift_data = sales_table_filt.shift(i)
        attr_name = 'item_cnt_day_shift_{}'.format(i)
        unstack_data = shift_data.unstack()
        attr_shift_data = unstack_data.reset_index().drop(columns = ['level_0']).rename(columns = {0:attr_name})
        data_join = data.merge(attr_shift_data, on = ['shop_id', 'item_id', 'date_block_num'], how = 'left')
    
    return data_join