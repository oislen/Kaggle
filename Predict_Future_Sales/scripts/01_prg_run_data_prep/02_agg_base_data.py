# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 20:33:00 2020

@author: oislen
"""

import pandas as pd
import reference.clean_constants as clean_cons
import reference.utilities as utl

def agg_base_data(cons):

    """
    """
        
    print('Loading clean data ...')
    
    # load in the raw data
    item_categories, items, sales_train, sample_submission, shops, test = utl.load_files('clean', cons)
    
    print('aggregating base data ...')
    
    # want to aggregate to year, month, shop and product level
    agg_base = sales_train.groupby(clean_cons.group_cols, as_index = False).agg(clean_cons.agg_dict)
    
    # add ID column
    agg_base['ID'] = agg_base.index
    
    print('Create generalised test data ...')
    
    # create static columns
    test['year'] = 2015
    test['month'] = 11
    test['date_block_num'] = 34
    test['item_cnt_day'] = -999
    test['n_refund'] = -999
    test['n_sale'] = -999

    # Generate most recent item price for test set 
    recent_price = utl.gen_most_recent_item_price(dataset = agg_base)
    join_cols = ['item_id']
    base_test_price = test.merge(recent_price, on = join_cols, how = 'left')
    
    # Fill in -999 default for missing prices 
    base_test_price['item_price'] = base_test_price['item_price'].fillna(-999)

    print('Concatenate Base and Test data ...')
    
    base_concat = pd.concat(objs = [agg_base, base_test_price], axis = 0, ignore_index = True)
    
    # data shape
    shape = base_concat.shape
    
    print('outputting aggregated base data {} ...'.format(shape))
    
    # output aggreated base data as feather file
    base_concat.to_feather(cons.base_agg_data_fpath)
    
    return
