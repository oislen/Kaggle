# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 17:46:31 2020

@author: oislen
"""

import pandas as pd
import file_constants as cons

def load_files(ver):
    
    """
    Loads in all files
    """
    
    if ver == 'raw':
        encoding = 'latin1'
        item_categories = pd.read_csv(cons.item_categories_fpath, encoding = encoding)
        items = pd.read_csv(cons.items_fpath)
        sales_train = pd.read_csv(cons.sales_train_fpath)
        sample_submission = pd.read_csv(cons.sample_submission_fpath) 
        shops = pd.read_csv(cons.shops_fpath, encoding = encoding)
        test = pd.read_csv(cons.test_fpath)
        
    elif ver == 'clean':
        
        items = pd.read_feather(cons.items_clean_fpath)
        sample_submission = pd.read_feather(cons.sample_submission_clean_fpath)
        sales_train = pd.read_feather(cons.sales_train_clean_fpath)
        test = pd.read_feather(cons.test_clean_fpath)
        item_categories = pd.read_feather(cons.item_categories_clean_fpath)
        shops = pd.read_feather(cons.shops_clean_fpath)
        
    return item_categories, items, sales_train, sample_submission, shops, test
    
