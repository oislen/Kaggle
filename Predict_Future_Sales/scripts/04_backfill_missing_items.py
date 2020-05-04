# -*- coding: utf-8 -*-
"""
Created on Mon May  4 16:47:19 2020

@author: oislen
"""

import pandas as pd
import file_constants as cons
import numpy as np

def back_fill_missing_items():
    
    """
    
    Back Fill Missing Items Documentation
    
    This function back fills items fold in the holdout set that are not found in the train, valid and test sets
    
    """
    agg_base = pd.read_feather(cons.base_agg_data_fpath)
    
    # find items which have no previous sales record
    holdout = agg_base[agg_base['data_split'] == 'holdout']
    filt_missing_price = holdout['item_price'].isnull()
    filt_missing_price.sum()
    
    # fill in -999s defaults for unknown prices
    holdout['item_price'] = holdout['item_price'].fillna(-999)
    holdout['price_decimal'] = holdout['price_decimal'].fillna(-999)
    holdout['price_decimal_len'] = holdout['price_decimal_len'].fillna(-999)
    
    sub_cols = ['shop_id', 'item_id', 'item_price', 'item_cnt_day', 'n_refund', 
                'n_sale', 'price_decimal', 'price_decimal_len',
                'item_name', 'item_category_id', 'item_category_name', 'item_cat', 'item_cat_sub',
                'shop_name', 'shop_quotes', 'shop_brackets', 'shop_smooth', 'data_split', 'ID',
                'n_weekenddays', 'n_publicholidays', 'totalholidays']
    
    missing_items = holdout.loc[filt_missing_price, sub_cols]

    missing_items['item_cnt_day'] = missing_items['item_cnt_day'].fillna(0)
    missing_items['n_refund'] = missing_items['n_refund'].fillna(0)
    missing_items['n_sale'] = missing_items['n_sale'].fillna(0)
    
    missing_items.isnull().sum()
    
    # extract missing item and shop ids
    miss_shop_ids = missing_items['shop_id']
    miss_item_ids = missing_items['item_id']
    
    # create an empty data frame to hold the data
    fill_data_df = pd.DataFrame()
    
    # join missing records
    for dbn in agg_base['date_block_num'].unique():
        print(dbn)
    
        # extract the dbn chunks
        dbn_chunk = agg_base.loc[agg_base['date_block_num'] == dbn, :]
        dbn_shop_ids = dbn_chunk['shop_id']
        dbn_item_ids = dbn_chunk['item_id']
        
        if dbn_chunk['data_split'].unique()[0] != 'holdout':
        
            # case where a shop has no record of item
            print(miss_shop_ids.isin(dbn_shop_ids).sum() / miss_shop_ids.shape[0])
            print(miss_item_ids.isin(dbn_item_ids).sum() / miss_item_ids.shape[0])
            
            filt_shop = miss_shop_ids.isin(dbn_shop_ids)
            missing_shop_items = missing_items.loc[filt_shop, :]
            missing_shop_items['year'] = dbn_chunk['year'].unique()[0]
            missing_shop_items['month'] = dbn_chunk['month'].unique()[0]
            missing_shop_items['date_block_num'] = dbn_chunk['date_block_num'].unique()[0]
            missing_shop_items['data_split'] = dbn_chunk['data_split'].unique()[0]
            missing_shop_items['n_weekenddays'] = dbn_chunk['n_weekenddays'].unique()[0]
            missing_shop_items['n_publicholidays'] = dbn_chunk['n_publicholidays'].unique()[0]
            missing_shop_items['totalholidays'] = dbn_chunk['totalholidays'].unique()[0]
            
            # concatenate the date month chunk with the missing shop items
            dbn_cat_chunk = pd.concat([dbn_chunk, missing_shop_items], axis = 0, ignore_index = True)
            
            # concatenate onto empty dataframe
            fill_data_df = pd.concat([fill_data_df, dbn_cat_chunk], axis = 0, ignore_index = True)
        
        else:
        
            # case where a shop has no record of item
            print(miss_shop_ids.isin(dbn_shop_ids).sum() / miss_shop_ids.shape[0])
            print(miss_item_ids.isin(dbn_item_ids).sum() / miss_item_ids.shape[0])
                    
            # concatenate onto empty dataframe
            fill_data_df = pd.concat([fill_data_df, holdout], axis = 0, ignore_index = True)
        
            
    fill_data_df = fill_data_df.sort_values(['year', 'month', 'shop_id', 'item_id']).reset_index(drop = True)
    
    # update ID column
    fill_data_df['ID'].value_counts(dropna = False)
    data_splits = ['train', 'valid', 'test']
    for split in data_splits:
        print(split)
        filt_split = fill_data_df['data_split'] == split
        n_train_obs = filt_split.sum()
        id_cols = ['ID']
        fill_data_df.loc[filt_split, id_cols] = range(n_train_obs)
        
    # output aggreated base data as feather file
    fill_data_df.to_feather(cons.base_agg_comp_fpath)
    
back_fill_missing_items()