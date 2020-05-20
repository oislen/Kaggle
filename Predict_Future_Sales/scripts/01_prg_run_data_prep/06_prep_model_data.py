# -*- coding: utf-8 -*-
"""
Created on Sun May 10 19:15:38 2020

@author: oislen
"""

import pandas as pd
import numpy as np
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
    
    # TODO: replace with mean encodings here
    
    # label encode  item cat
    item_cat_label_enc = preprocessing.LabelEncoder()
    item_cat_label_enc.fit(base['item_cat'].unique())
    base['item_cat_enc'] = item_cat_label_enc.transform(base['item_cat'])
    
    # label encode item cat sub
    item_cat_sub_label_enc = preprocessing.LabelEncoder()
    item_cat_sub_label_enc.fit(base['item_cat_sub'].unique())
    base['item_cat_sub_enc'] = item_cat_sub_label_enc.transform(base['item_cat_sub'])
    
    # label encode city
    city_cat_sub_label_enc = preprocessing.LabelEncoder()
    city_cat_sub_label_enc.fit(base['city'].unique())
    base['city_enc'] = city_cat_sub_label_enc.transform(base['city'])
    
    # set columns to drop
    drop_cols = ['item_cat_sub', 'item_cat', 'city']
    model_data = base.drop(columns = drop_cols)
    
    # TODO: add down casting here to float16 and int8
    
    """
    data_cols = model_data.columns
    data_dtypes = model_data.dtypes
    data_dtypes.value_counts()
    
    int32_cols = data_cols[data_dtypes == np.int32]
    int64_cols = data_cols[data_dtypes == np.int64]
    float32_cols = data_cols[data_dtypes == np.float64]
    
    model_data[int32_cols] = model_data[int32_cols].astype(np.int8)
    model_data[int64_cols] = model_data[int64_cols].astype(np.int8)
    model_data[float32_cols] = model_data[float32_cols].astype(np.float32)
    
    model_data.dtypes.value_counts()
    """

    shape = model_data.shape
    
    print('outputting model data {} ....'.format(shape))
    
    # output the model data
    model_data.to_feather(cons.model_data_fpath)

    return
