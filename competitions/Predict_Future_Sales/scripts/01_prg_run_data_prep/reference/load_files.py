# -*- coding: utf-8 -*-
"""
Created on Sun May  9 15:29:40 2021

@author: oislen
"""

import pandas as pd

def load_files(ver, cons):
    
    """
    
    Load Files Documentation
    
    Function Overview
    
    This function loads in either the raw or cleaned data.
    The raw data is loaded as a .csv file with latin1 encoding.
    The clean data is loaded as a .feather file.
    
    Defaults
    
    load_files(ver, 
               cons
               )
    
    Parameters
    
    ver - String, the data to be loaded in, either raw or clean.
    cons - Python Module, the programme constants
    
    Returns
    
    item_categories - DataFrame, the item categories data
    items - DataFrame, the items data 
    sales_train - DataFrame, the sales data for training 
    sample_submission - DataFrame, the sample submission file 
    shops - DataFrame, the shops data 
    test - DataFrame, the test data
    
    Example
    
    load_files(ver = 'raw', 
               cons = cons
               )
    
    """
    
    # if loading raw data
    if ver == 'raw':
        
        # set encoding
        encoding = 'latin1'
        
        # load in datasets from .csv files
        item_categories = pd.read_csv(cons.item_categories_fpath, encoding = encoding)
        items = pd.read_csv(cons.items_fpath)
        sales_train = pd.read_csv(cons.sales_train_fpath)
        sample_submission = pd.read_csv(cons.sample_submission_fpath) 
        shops = pd.read_csv(cons.shops_fpath, encoding = encoding)
        test = pd.read_csv(cons.test_fpath)
        
    # else if loading in the cleaned data
    elif ver == 'clean':
        
        # load in the datasets from .feather files
        items = pd.read_feather(cons.items_clean_fpath)
        sample_submission = pd.read_feather(cons.sample_submission_clean_fpath)
        sales_train = pd.read_feather(cons.sales_train_clean_fpath)
        test = pd.read_feather(cons.test_clean_fpath)
        item_categories = pd.read_feather(cons.item_categories_clean_fpath)
        shops = pd.read_feather(cons.shops_clean_fpath)
        
    return item_categories, items, sales_train, sample_submission, shops, test
    
