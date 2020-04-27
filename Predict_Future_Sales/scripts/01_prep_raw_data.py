# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 20:58:03 2020

@author: oislen
"""

import pandas as pd
import file_constants as cons
import utilities as utl

def prep_raw_data():
    
    """
    """
    
    # load in the raw data
    item_categories, items, sales_train, sample_submission, shops, test = utl.load_files('raw')
    
    #-- Sales Data --#
    
    print('Preparing Sales Data ...')
    
    # prep sales train data
    sales_train['date'] = pd.to_datetime(sales_train['date'], format = '%d.%m.%Y')
    sales_train['day'] = sales_train['date'].dt.day
    sales_train['month'] = sales_train['date'].dt.month
    sales_train['year'] = sales_train['date'].dt.year
    
    # extract out the sales and refunds
    sale_lam = lambda x: 0 if x <= 0 else x
    ref_lam = lambda x: 0 if x >= 0 else x
    sales_train['n_refund'] = sales_train['item_cnt_day'].apply(ref_lam)
    sales_train['n_sale'] = sales_train['item_cnt_day'].apply(sale_lam)
    
    # extract price information
    sales_train['price_decimal'] = sales_train['item_price'].astype(str).str.extract('\d+\.(\d*)')[0].astype(float)
    sales_train['price_decimal_len'] = sales_train['item_price'].astype(str).str.extract('\d+\.(\d*)')[0].str.len()
    
    #-- Item Categories --#
    
    print('Preparing Item Data ...')
    
    # extract the item category and sub-category
    item_categories['item_category_name'].value_counts()
    item_categories['item_cat'] = item_categories['item_category_name'].str.split(' - ', expand = True)[0]
    item_categories['item_cat_sub'] = item_categories['item_category_name'].str.split(' - ', expand = True)[1]
   
    #-- Shop Name --#
    
    print('Preparing Shop Data ...')
    
    shops['shop_quotes'] = shops['shop_name'].str.extract('(".*")')[0]
    shops['shop_brackets'] = shops['shop_name'].str.extract('(\(.*\))')[0]
    shops['shop_smooth'] = shops['shop_name'].str.replace('".*"|\(.*\)', "")
    
    #-- Output Files --#
    
    # output the data
    sample_submission.to_feather(cons.sample_submission_clean_fpath)
    items.to_feather(cons.items_clean_fpath)
    shops.to_feather(cons.shops_clean_fpath)
    item_categories.to_feather(cons.item_categories_clean_fpath)
    sales_train.to_feather(cons.sales_train_clean_fpath)
    test.to_feather(cons.test_clean_fpath)
    
    return

prep_raw_data()