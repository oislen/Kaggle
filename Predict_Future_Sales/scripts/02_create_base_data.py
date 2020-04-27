# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 20:11:27 2020

@author: oislen
"""

import utilities as utl
import pandas as pd
import file_constants as cons

pd.set_option('display.max_columns', 20)

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
 
    print('Outputting Base data ...')
    
    # output the base data
    sales_items_cat_shop.to_feather(cons.base_raw_data_fpath)
    
    return 
    
create_base_data()