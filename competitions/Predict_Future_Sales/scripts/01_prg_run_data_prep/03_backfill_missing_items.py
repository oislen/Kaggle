# -*- coding: utf-8 -*-
"""
Created on Sat May  9 19:23:38 2020

@author: oislen
"""

# import relevant libraries
import pandas as pd
from reference.load_files import load_files
from reference.backfill_attr import backfill_attr
from reference.fill_id import fill_id
from reference.recast_df import recast_df

def back_fill_missing_items(cons):
    
    """
    
    Back Fill Missing Items Documentation
    
    Function Overview
    
    This function back fills items found in the holdout set that are not found in the train, valid and test sets
    Additional process also include creating a primary key, creating indicators for missing data and irrelevant obserations.
    
    Defaults
    
    back_fill_missing_items(cons)
    
    Parameters
    
    cons - Python Module, the programme constants for the competition
    
    Returns
    
    0 for successful execution
    
    Example
    
    back_fill_missing_items(cons)
    
    """
    
    print('working on item sells and price ...')
    
    # load in the raw data
    item_categories, items, sales_train, sample_submission, shops, test = load_files('clean', cons)
    
    agg_base = pd.read_feather(cons.base_agg_data_fpath)
    
    del sales_train, sample_submission, test
    
    #-- Price --#
    
    price_unstack = backfill_attr(dataset = agg_base, 
                                  pivot_values = ['item_price'], 
                                  fillna = -999,
                                  pivot_index = ['date_block_num'], 
                                  pivot_columns = ['shop_id', 'item_id'], 
                                  ffill = True
                                  )
    
    #-- Monthly Sales --#
    
    total_unstack = backfill_attr(dataset = agg_base, 
                                  pivot_values = ['item_cnt_day'], 
                                  fillna = 0,
                                  pivot_index = ['date_block_num'], 
                                  pivot_columns = ['shop_id', 'item_id'], 
                                  ffill = False
                                  )
    
    #-- Renenue --#
    
    revenue_unstack = backfill_attr(dataset = agg_base, 
                                    pivot_values = ['revenue'], 
                                    fillna = 0,
                                    pivot_index = ['date_block_num'], 
                                    pivot_columns = ['shop_id', 'item_id'], 
                                    ffill = False
                                    )
    
    #-- ID --#
    
    print('Subsetting a ID column ...')
    
    sub_cols = ['shop_id', 'item_id', 'date_block_num', 'ID']
    id_df = agg_base[sub_cols]
    
    del agg_base
    
    #-- Join Datasets --#

    print('Joining datasets ...')
    
    # create an empty df to join on to
    join_cols = ['date_block_num', 'shop_id', 'item_id']
    join_df = price_unstack[join_cols]
    join_df = join_df.merge(price_unstack, on = join_cols, how = 'left')
    join_df = join_df.merge(total_unstack, on = join_cols, how = 'left')
    join_df = join_df.merge(revenue_unstack, on = join_cols, how = 'left')
    join_df = join_df.merge(id_df, on = join_cols, how = 'left')
    
    #del price_unstack, total_unstack, refund_unstack, sales_unstack, id_df
    del price_unstack, total_unstack, id_df

    print('Adding data set splits ...')
    
    join_df['data_split'] = join_df['date_block_num'].apply(lambda x: 'train' if x  <= 33 else 'holdout')
    join_df['meta_level'] = join_df['date_block_num'].apply(lambda x: 'level_1' if x  <= 29 else ('level_2' if x >= 30 and x <= 33 else 'holdout'))
    
    print('Filling in ID ...')
    
    join_df = fill_id(dataset = join_df, fill_type = 'range', split = 'train')
    join_df = fill_id(dataset = join_df, fill_type = 'range', split = 'valid')
    join_df = fill_id(dataset = join_df, fill_type = 'range', split = 'test')
    join_df = fill_id(dataset = join_df, fill_type = 'value', split = 'holdout', fillna = -999)
    
    print('Create primary key ...')
    
    join_df['primary_key'] = join_df.index
    
    print('Create holdout subset indicator ...')
    
    filt_holdout = join_df['data_split'] == 'holdout'
    filt_id = join_df['ID'] != -999
    join_df['holdout_subset_ind'] = (filt_id & filt_holdout).astype(int)
    
    print('Mapping missing holdout sales info ...')
    
    join_df.loc[filt_holdout, 'item_cnt_day'] = 0
    
    print('Create no sales history indicator ...')
    
    filt_default_price = join_df['item_price'] == -999
    join_df['no_sales_hist_ind'] = filt_default_price.astype(int)
    
    print('Create no sales history holdout set indicator ...')
    
    filt_no_sales_holdout = filt_default_price & filt_holdout
    join_df['no_holdout_sales_hist_ind'] = filt_no_sales_holdout.astype(int)
    join_df['item_price'] = join_df['item_price'].replace(-999, 0)
    
    print('Removing observations not in holdout set ...')
    
    # NOTE: this step drops a lot of information
    # need to filter out excess items not found n holdout set
    # ideally this should save our runtime resources
    join_df['shop_item_id'] = join_df['shop_id'].astype(str) + '_' + join_df['item_id'].astype(str)
    holdout = join_df[join_df['data_split'] == 'holdout']
    id_null = holdout['ID'] == -999
    null_holdout = holdout[id_null]
    shop_item_id = null_holdout['shop_item_id'].unique()
    filt_no_test = ~join_df['shop_item_id'].isin(shop_item_id)
    join_df_filt = join_df.loc[filt_no_test, :].reset_index(drop = True)
    join_df_filt['data_split'].value_counts() 
    join_df['data_split'].value_counts() 

    print('Recast data ...')
    
    join_df_filt = recast_df(dataset = join_df_filt)
    
    shape = join_df_filt.shape

    print('Outputting file {} ...'.format(shape))
    
    # output aggreated base data as feather file
    join_df_filt.to_feather(cons.base_agg_comp_fpath)
    
    return 0
