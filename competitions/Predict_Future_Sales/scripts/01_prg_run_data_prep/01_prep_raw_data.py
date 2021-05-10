# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 20:58:03 2020

@author: oislen
"""

import pandas as pd
from reference.load_files import load_files
from reference.recast_df import recast_df
from sklearn import preprocessing

def prep_raw_data(cons):
    
    """
    
    Prepare Raw Data Documentation
    
    Function Overview
    
    This function prepares and cleans each individual raw dataset.
    The prepped / cleaned raw dataset is output as a feather file for the next step in the pipeline.
    
    Defaults
    
    prep_raw_data()
    
    Parameters
    
    Returns
    
    Outputs
    
    Example
    
    prep_raw_data()
    
    """
    
    # load in the raw data
    item_categories, items, sales_train, sample_submission, shops, test = load_files('raw', cons)
    
    #-- Sales Data --#
    
    print('Preparing Sales Data ...')
    
    print('Extracting date information ...')
    # prep sales train data
    sales_train['date'] = pd.to_datetime(sales_train['date'], format = '%d.%m.%Y')
    sales_train['day'] = sales_train['date'].dt.day
    sales_train['month'] = sales_train['date'].dt.month
    sales_train['year'] = sales_train['date'].dt.year
    
    print('Capping price and sales count ...')
    sales_train['item_cnt_day'] = sales_train['item_cnt_day'].apply(lambda x: x if x < 1000 else 1000)
    sales_train['item_price'] = sales_train['item_price'].apply(lambda x: x if x < 100000 else 100000)
    
    print('Extracting sales and refund information ...')
    # extract out the sales and refunds
    sale_lam = lambda x: 0 if x <= 0 else x
    ref_lam = lambda x: 0 if x >= 0 else -1 * x
    sales_train['n_refund'] = sales_train['item_cnt_day'].apply(ref_lam)
    sales_train['n_sale'] = sales_train['item_cnt_day'].apply(sale_lam)
    
    print('Fill negative item price with median ...')    
    neg_item_price = sales_train['item_price'] < 0
    sales_train.loc[neg_item_price, 'item_price'] = sales_train['item_price'].median()
    
    #-- Item Categories --#
    
    print('Preparing Item Data ...')
    
    print('Extracting category information ...')
    # extract the item category and sub-category
    item_categories['item_category_name'].value_counts()
    item_cat = item_categories['item_category_name'].str.split(' - ', expand = True)[0]
    item_cat_sub = item_categories['item_category_name'].str.split(' - ', expand = True)[1]
        
    # clean item categories
    item_cat = item_cat.str.replace('Payment.*', 'Payment Cards')
    item_categories['item_cat'] = item_cat.str.replace('.*[Gg]ames.*', 'Games')
        
    # clean item cat
    item_cat_sub = item_cat_sub.fillna('')
    item_cat_sub = item_cat_sub.str.replace('^Audiobooks.*', 'Audiobooks')
    item_cat_sub = item_cat_sub.str.replace('^CD.*', 'CD')
    item_cat_sub = item_cat_sub.str.replace('^Live.*', 'Live')
    item_categories['item_cat_sub'] = item_cat_sub.str.replace('^Teaching.*', 'Teaching')
    
    # label encode item cat
    item_cat_label_enc = preprocessing.LabelEncoder()
    item_cat_label_enc.fit(item_categories['item_cat'].unique())
    item_categories['item_cat_id'] = item_cat_label_enc.transform(item_categories['item_cat'])
    
    # label encode item cat sub
    item_cat_sub_label_enc = preprocessing.LabelEncoder()
    item_cat_sub_label_enc.fit(item_categories['item_cat_sub'].unique())
    item_categories['item_cat_sub_id'] = item_cat_sub_label_enc.transform(item_categories['item_cat_sub'])
    
    #-- Shop Name --#
    
    print('Preparing Shop Data ...')
    
    shop_filt = shops['shop_name'] == 'Сергиев Посад ТЦ "7Я"'
    shops.loc[shop_filt, 'shop_name'] = 'СергиевПосад ТЦ "7Я"'
    shops['city'] = shops['shop_name'].str.split(' ').map(lambda x: x[0])
    city_filt = shops['city'] == '!Якутск'
    shops.loc[city_filt, 'city'] = 'Якутск'

    # label encode city
    city_cat_sub_label_enc = preprocessing.LabelEncoder()
    city_cat_sub_label_enc.fit(shops['city'].unique())
    shops['city_enc'] = city_cat_sub_label_enc.transform(shops['city'])
    
    #-- Downcase Data --#
    
    # recast data
    sample_submission = recast_df(dataset = sample_submission)
    items = recast_df(dataset = items)
    shops = recast_df(dataset = shops)
    item_categories = recast_df(dataset = item_categories)
    sales_train = recast_df(dataset = sales_train)
    test = recast_df(dataset = test)
    
    #-- Output Files --#
    
    print('Outputting cleaned raw data ...')
    
    # output the data
    sample_submission.to_feather(cons.sample_submission_clean_fpath)
    items.to_feather(cons.items_clean_fpath)
    shops.to_feather(cons.shops_clean_fpath)
    item_categories.to_feather(cons.item_categories_clean_fpath)
    sales_train.to_feather(cons.sales_train_clean_fpath)
    test.to_feather(cons.test_clean_fpath)
    
    return
