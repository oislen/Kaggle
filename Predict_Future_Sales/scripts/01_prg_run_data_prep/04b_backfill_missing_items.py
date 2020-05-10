# -*- coding: utf-8 -*-
"""
Created on Sat May  9 19:23:38 2020

@author: oislen
"""

import pandas as pd
import reference.file_constants as cons
import reference.utilities as utl

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
    
    price_unstack = utl.backfill_attr(dataset = agg_base, 
                                      pivot_values = ['item_price'], 
                                      fillna = -999,
                                      pivot_index = ['year', 'month'], 
                                      pivot_columns = ['shop_id', 'item_id'], 
                                      ffill = True
                                      )
    
    #-- Total --#
    
    total_unstack = utl.backfill_attr(dataset = agg_base, 
                                      pivot_values = ['item_cnt_day'], 
                                      fillna = 0,
                                      pivot_index = ['year', 'month'], 
                                      pivot_columns = ['shop_id', 'item_id'], 
                                      ffill = False
                                      )
    
    #-- Refund --#
    
    refund_unstack = utl.backfill_attr(dataset = agg_base, 
                                       pivot_values = ['n_refund'], 
                                       fillna = 0,
                                       pivot_index = ['year', 'month'], 
                                       pivot_columns = ['shop_id', 'item_id'], 
                                       ffill = False
                                       )

    #-- Sales --#
    
    sales_unstack = utl.backfill_attr(dataset = agg_base, 
                                      pivot_values = ['n_sale'], 
                                      fillna = 0,
                                      pivot_index = ['year', 'month'], 
                                      pivot_columns = ['shop_id', 'item_id'], 
                                      ffill = False
                                      )
    
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
    
    print('Adding data set splits ...')
    
    join_df['data_split'] = join_df['date_block_num'].apply(lambda x: 'train' if x  <= 31 else ('valid' if x == 32 else ('test' if x == 33 else 'holdout')))
    
    print('Filling in ID ...')
    
    join_df = utl.fill_id(dataset = join_df, fill_type = 'range', split = 'train')
    join_df = utl.fill_id(dataset = join_df, fill_type = 'range', split = 'valid')
    join_df = utl.fill_id(dataset = join_df, fill_type = 'range', split = 'test')
    join_df = utl.fill_id(dataset = join_df, fill_type = 'value', split = 'holdout', fillna = -999)
    
    print('Create primary key ...')
    
    join_df['primary_key'] = join_df.index

    print('Outputting file ...')
    
    # output aggreated base data as feather file
    join_df.to_feather(cons.base_agg_comp_fpath)
    
    return 
