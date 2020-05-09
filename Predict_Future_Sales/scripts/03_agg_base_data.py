# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 20:33:00 2020

@author: oislen
"""

import pandas as pd
import file_constants as cons
import clean_constants as clean_cons
import utilities as utl

def agg_base_data():

    """
    """
    
    print('loading base data ...')
    
    # load in the raw data
    item_categories, items, sales_train, sample_submission, shops, test = utl.load_files('clean')
    
    # load in base data
    base_raw = pd.read_feather(cons.base_raw_data_fpath)
    base_test = pd.read_feather(cons.base_raw_test_fpath)
    
    print('aggregating base data ...')
    
    # want to aggregate to year, month, shop and product level
    agg_base = base_raw.groupby(clean_cons.group_cols, as_index = False).agg(clean_cons.agg_dict)
    
    # add ID column
    agg_base['ID'] = agg_base.index
    
    print('Generate most recent item price for test set ...')

    recent_price = utl.gen_most_recent_item_price(dataset = agg_base)
    join_cols = ['item_id']
    base_test_price = base_test.merge(recent_price, on = join_cols, how = 'left')
    base_test_price['item_price'] = base_test_price['item_price'].fillna(-999)

    print('Concatenate Base and Test data ...')
    
    base_concat = pd.concat(objs = [agg_base, base_test_price], axis = 0, ignore_index = True)
    
    print('outputting aggregated base data ...')
    
    # output aggreated base data as feather file
    base_concat.to_feather(cons.base_agg_data_fpath)
    
    return

agg_base_data()