# -*- coding: utf-8 -*-
"""
Created on Sat May  9 19:23:38 2020

@author: oislen
"""

import pandas as pd
import file_constants as cons
import utilities as utl

def back_fill_missing_items():
    
    """
    
    Back Fill Missing Items Documentation
    
    This function back fills items fold in the holdout set that are not found in the train, valid and test sets
    
    """
    
    # load in the raw data
    item_categories, items, sales_train, sample_submission, shops, test = utl.load_files('clean')
    
    agg_base = pd.read_feather(cons.base_agg_data_fpath)
    
    del sales_train, sample_submission, test
    
    #-- Price --#
    
    print('Back filling price ...')
    
    # set up for finding price of an item
    price_table = pd.pivot_table(data = agg_base, 
                                 values = ['item_price'],
                                 index = ['year', 'month'],
                                 columns = ['shop_id', 'item_id'],
                                 dropna = False
                                 )

    # want to forward fill from given price to next price
    price_table_ff = price_table.ffill(axis = 0)
    price_table_fill = price_table_ff.fillna(-999)
    
    # double unstack to get data by year, month, item_id and shop_id
    price_unstack = price_table_fill.stack().stack().reset_index()
    
    # filter out future dates
    date_filt = (price_unstack['year'] == 2015) & (price_unstack['month'] == 12)
    price_unstack = price_unstack.loc[~date_filt, :]
    
    del price_table
    
    #-- Total --#
    
    print('Back fill total ...')
    
    # set up for finding price of an item
    total_table = pd.pivot_table(data = agg_base, 
                                 values = ['item_cnt_day'],
                                 index = ['year', 'month'],
                                 columns = ['shop_id', 'item_id'],
                                 dropna = False
                                 )
    
    # double unstack to get data by year, month, item_id and shop_id
    total_unstack = total_table.fillna(0).stack().stack().reset_index()
    
    # filter out future dates
    date_filt = (total_unstack['year'] == 2015) & (total_unstack['month'] == 12)
    total_unstack = total_unstack.loc[~date_filt, :]
    
    del total_table
    
    #-- Refund --#
    
    print('Back fill refund ...')
    
    # set up for finding price of an item
    refund_table = pd.pivot_table(data = agg_base, 
                                 values = ['n_refund'],
                                 index = ['year', 'month'],
                                 columns = ['shop_id', 'item_id'],
                                 dropna = False
                                 )
    
    # double unstack to get data by year, month, item_id and shop_id
    refund_unstack = refund_table.fillna(0).stack().stack().reset_index()
    
    # filter out future dates
    date_filt = (refund_unstack['year'] == 2015) & (refund_unstack['month'] == 12)
    refund_unstack = refund_unstack.loc[~date_filt, :]
    
    del refund_table
    
    #-- Sales --#
    
    print('Back fill sales ...')
    
    # set up for finding price of an item
    sales_table = pd.pivot_table(data = agg_base, 
                                 values = ['n_sale'],
                                 index = ['year', 'month'],
                                 columns = ['shop_id', 'item_id'],
                                 dropna = False
                                 )
    
    # double unstack to get data by year, month, item_id and shop_id
    sales_unstack = sales_table.fillna(0).stack().stack().reset_index()
      
    # filter out future dates
    date_filt = (sales_unstack['year'] == 2015) & (sales_unstack['month'] == 12)
    sales_unstack = sales_unstack.loc[~date_filt, :]
      
    del sales_table
    
    #-- ID --#
    
    print('Subsetting a ID column ...')
    
    sub_cols = ['year', 'month', 'shop_id', 'item_id', 'date_block_num', 'ID']
    id_df = agg_base[sub_cols]
    
    del agg_base
    
    #-- Join Datasets --#

    print('Joining datasets ...')
    
    # create an empty df to join on to
    join_df = price_unstack[['year', 'month', 'shop_id', 'item_id']]
    join_df = join_df.merge(price_unstack, on = ['year', 'month', 'shop_id', 'item_id'], how = 'left')
    join_df = join_df.merge(total_unstack, on = ['year', 'month', 'shop_id', 'item_id'], how = 'left')
    join_df = join_df.merge(refund_unstack, on = ['year', 'month', 'shop_id', 'item_id'], how = 'left')
    join_df = join_df.merge(sales_unstack, on = ['year', 'month', 'shop_id', 'item_id'], how = 'left')
    join_df = join_df.merge(id_df, on = ['year', 'month', 'shop_id', 'item_id'], how = 'left')
    
    del price_unstack, total_unstack, refund_unstack, sales_unstack, id_df
    
    print('Fill date block ...')
    
    join_df['date_block_num'] = join_df['date_block_num'].ffill().bfill()
    
    print('Extract the price info ...')
    
    # extract price information
    join_df['price_decimal'] = join_df['item_price'].astype(str).str.extract('\d+\.(\d*)')[0].astype(float)
    join_df['price_decimal_len'] = join_df['item_price'].astype(str).str.extract('\d+\.(\d*)')[0].str.len()
    
    print('Joining clean data ...')
    
    # join all data sets together
    join_df = join_df.merge(items, on = 'item_id', how = 'left')
    join_df = join_df.merge(item_categories, on = 'item_category_id', how = 'left')
    join_df = join_df.merge(shops, on = 'shop_id', how = 'left')
    
    print('Subset required columns ...')
    
    sub_cols = ['year', 'month', 'shop_id', 'item_id', 'item_price', 'item_cnt_day', 'n_refund', 'n_sale', 'date_block_num', 'ID', 'price_decimal', 'price_decimal_len', 'item_category_id', 'item_cat', 'item_cat_sub']
    join_df = join_df[sub_cols]
    
    print('Generate calendar days ...')
    
    retail_calander = utl.gen_retail_calender()
    join_cols = ['year', 'month']
    join_df = join_df.merge(retail_calander, on = join_cols, how = 'left')
    
    print('Adding data set splits ...')
    
    join_df['data_split'] = join_df['date_block_num'].apply(lambda x: 'train' if x  <= 31 else ('valid' if x == 32 else ('test' if x == 33 else 'holdout')))
    
    
    # TODO: need to filter out excess items not found n holdout set
    
    print('Outputting file ...')
    
    # output aggreated base data as feather file
    join_df.to_feather(cons.base_agg_comp_fpath)
    
    return join_df

join_df = back_fill_missing_items()