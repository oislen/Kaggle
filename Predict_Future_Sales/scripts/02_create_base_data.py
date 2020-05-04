# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 20:11:27 2020

@author: oislen
"""

import utilities as utl
import pandas as pd
import numpy as np
import file_constants as cons

pd.set_option('display.max_columns', 30)

def create_base_data():
    
    """
    Create base data 
    """
    
    print('Loading clean data ...')
    
    # load in the raw data
    item_categories, items, sales_train, sample_submission, shops, test = utl.load_files('clean')
    
    # print column names
    sales_train.columns
    items.columns
    item_categories.columns
    shops.columns
    
    print('Joining clean data ...')
    
    # join all data sets together
    sales_items = sales_train.merge(items, on = 'item_id', how = 'left')
    sales_items_cat = sales_items.merge(item_categories, on = 'item_category_id', how = 'left')
    sales_items_cat_shop = sales_items_cat.merge(shops, on = 'shop_id', how = 'left')
    
    print('Create generalised test data ...')
    
    # create static columns
    test['year'] = 2015
    test['month'] = 11
    test['date_block_num'] = 34
    test['item_cnt_day'] = np.nan
    test['n_refund'] = np.nan
    test['n_sale'] = np.nan
    
    # join on other reference data
    test_items = test.merge(items, on = 'item_id', how = 'left')
    test_items_cat = test_items.merge(item_categories, on = 'item_category_id', how = 'left')
    test_items_cat_shop = test_items_cat.merge(shops, on = 'shop_id', how = 'left')
    
    print('Adding data set splits ...')
    
    sales_items_cat_shop['data_split'] = sales_items_cat_shop['date_block_num'].apply(lambda x: 'train' if x  <= 31 else ('valid' if x == 32 else 'test'))
    test_items_cat_shop['data_split'] = 'holdout'
    
    print('Outputting Base and Test data ...')
    
    # output the base data
    sales_items_cat_shop.to_feather(cons.base_raw_data_fpath)
    test_items_cat_shop.to_feather(cons.base_raw_test_fpath)
    
    return 
    
create_base_data()