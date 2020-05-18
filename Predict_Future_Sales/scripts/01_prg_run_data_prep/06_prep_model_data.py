# -*- coding: utf-8 -*-
"""
Created on Sun May 10 19:15:38 2020

@author: oislen
"""

import pandas as pd
from sklearn import preprocessing

def prep_model_data(cons):
    
    """
    
    Prepare Model Data Documentation
    
    Function Overview
    
    """
    
    print('loading in base data ...')
    
    # load in the bases data file
    base = pd.read_feather(cons.base_agg_supp_fpath)
    
    print('label encoding categorical variables ...')
    
    # label encode  item cat
    item_cat_label_enc = preprocessing.LabelEncoder()
    item_cat_label_enc.fit(base['item_cat'].unique())
    base['item_cat_enc'] = item_cat_label_enc.transform(base['item_cat'])
    
    # label encode item cat sub
    item_cat_sub_label_enc = preprocessing.LabelEncoder()
    item_cat_sub_label_enc.fit(base['item_cat_sub'].unique())
    base['item_cat_sub_enc'] = item_cat_sub_label_enc.transform(base['item_cat_sub'])
    
    # set columns to drop
    drop_cols = ['item_cat_sub', 'item_cat']
    model_data = base.drop(columns = drop_cols)
    
    shape = model_data.shape
    
    print('outputting model data {} ....'.format(shape))
    
    # output the model data
    model_data.to_feather(cons.model_data_fpath)

    return
