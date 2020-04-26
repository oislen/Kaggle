# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 17:46:31 2020

@author: oislen
"""

import pandas as pd
import file_constants as cons

def load_raw_files():
    """
    Loads in all raw files
    """
    item_categories = pd.read_csv(cons.item_categories_fpath)
    items = pd.read_csv(cons.items_fpath)
    sales_train = pd.read_csv(cons.sales_train_fpath)
    sample_submission = pd.read_csv(cons.sample_submission_fpath) 
    shops = pd.read_csv(cons.shops_fpath)
    test = pd.read_csv(cons.test_fpath)
    return item_categories, items, sales_train, sample_submission, shops, test
