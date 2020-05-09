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
    
    print('Create generalised test data ...')
    
    # create static columns
    test['year'] = 2015
    test['month'] = 11
    test['date_block_num'] = 34
    test['item_cnt_day'] = np.nan
    test['n_refund'] = np.nan
    test['n_sale'] = np.nan
    
    print('Outputting Base and Test data ...')
    
    # output the base data
    sales_train.to_feather(cons.base_raw_data_fpath)
    test.to_feather(cons.base_raw_test_fpath)
    
    return 
    
create_base_data()