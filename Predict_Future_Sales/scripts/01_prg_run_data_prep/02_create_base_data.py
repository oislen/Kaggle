# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 20:11:27 2020

@author: oislen
"""

import pandas as pd
import reference.utilities as utl
import reference.file_constants as cons

def create_base_data():
    
    """
    Create base data 
    """
    
    print('Loading clean data ...')
    
    # load in the raw data
    item_categories, items, sales_train, sample_submission, shops, test = utl.load_files('clean')
    
    print('Create generalised test data ...')
    
    # create static columns
    test['year'] = 2015
    test['month'] = 11
    test['date_block_num'] = 34
    test['item_cnt_day'] = -999
    test['n_refund'] = -999
    test['n_sale'] = -999
    
    print('Outputting Base and Test data ...')
    
    # output the base data
    sales_train.to_feather(cons.base_raw_data_fpath)
    test.to_feather(cons.base_raw_test_fpath)
    
    return 
    