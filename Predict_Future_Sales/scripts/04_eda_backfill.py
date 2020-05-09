# -*- coding: utf-8 -*-
"""
Created on Mon May  4 16:57:06 2020

@author: oislen
"""



import file_constants as cons
import pandas as pd
import utilities as utl

agg_base = pd.read_feather(cons.base_agg_comp_fpath)
agg_base = pd.read_feather(cons.base_agg_data_fpath)

agg_base['item_id'].value_counts()
agg_base['shop_id'].value_counts()
agg_base['item_cnt_day'].value_counts()[0:20]
(agg_base['item_id'].astype(str) + '_' + agg_base['shop_id'].astype(str)).value_counts()

group_cols = ['item_id']
agg_base.groupby(group_cols).agg()
agg_base['date_block_num'].unique()

dbn = 1

dbn_chunks = agg_base[agg_base['date_block_num'] == dbn]
dbn_chunks['item_id'].value_counts()
dbn_chunks.isnull().sum()

train = agg_base[agg_base['data_split'] == 'train']
valid = agg_base[agg_base['data_split'] == 'valid']
test = agg_base[agg_base['data_split'] == 'test']
holdout = agg_base[agg_base['data_split'] == 'holdout']


holdout['item_price'].value_counts(dropna = False)
(holdout['item_price'] == -999).sum()

#-- Item Price --#

# set up for finding price of an item
price_table = pd.pivot_table(data = agg_base, 
                             values = ['item_price'],
                             index = ['year', 'month'],
                             columns = ['shop_id', 'item_id'],
                             #fill_value = 0,
                             dropna = False
                             )

(price_table == -999).any().any()
price_table.isnull().all().any()

# want to forward fill from given price to next price
price_table_ff = price_table.ffill(axis = 0)
price_table_fill = price_table_ff.fillna(-999)

# double unstack to get data by year, month, item_id and shop_id
price_unstack = price_table_fill.stack().stack().reset_index()

#-- Quantity Sold --#

# set up for finding price of an item
sales_table = pd.pivot_table(data = agg_base, 
                             values = ['item_cnt_day'],
                             index = ['year', 'month'],
                             columns = ['shop_id', 'item_id'],
                             fill_value = 0,
                             dropna = False
                             )

# double unstack to get data by year, month, item_id and shop_id
sales_unstack = sales_table.stack().stack().reset_index()

# join data together
if price_unstack.shape == sales_unstack.shape:
    
    join = pd.merge(left = sales_unstack, 
                    right = price_unstack,
                    on = ['year', 'month', 'shop_id', 'item_id'],
                    how = 'left'
                    )
    
    join
    
join['item_cnt_day'].value_counts()
join['item_price'].value_counts()
