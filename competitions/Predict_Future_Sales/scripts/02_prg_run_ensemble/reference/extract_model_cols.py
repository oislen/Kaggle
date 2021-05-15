# -*- coding: utf-8 -*-
"""
Created on Sat May 15 08:31:37 2021

@author: oislen
"""

def extract_model_cols(dataset):
    
    """
    
    Extract Model Columns Documentation
    
    Function Overview
    
    This function splits up the columns of a given modeling dataset into index columns, target columns, excluded columns and predictor columns.
    The results are returned as arrays within a python dictionary.
    
    Defaults
    
    extract_model_cols(dataset)
    
    Parameters
    
    dataset - DataFrame, the modelling dataset to extract the various columns from.
    
    Returns
    
    model_cols_dict - Dictionary, the extract columnsstored within arrays
    
    Example
    
    extract_model_cols(dataset = base)
    
    """
    
    print('extracting out dataset columns ...')
    
    # extract the dataset columns
    data_cols = dataset.columns.tolist()
    
    # seperate out the index, target and predictor columns
    index_cols = ['primary_key', 
                  'ID', 
                  'data_split',
                  'meta_level',
                  'holdout_subset_ind',
                  'no_sales_hist_ind', 
                  'no_holdout_sales_hist_ind'
                  ]
    
    tar_cols = ['item_cnt_day']
    
    # the columns below contain information which would forward bias the results (data leakage from target)
    excl_cols = ['shop_id_total_item_cnt_day', 
                 'item_id_total_item_cnt_day',
                 'item_category_id_total_item_cnt_day',
                 'shop_id_item_category_id_total_item_cnt_day',
                 'city_enc_total_item_cnt_day',
                 'item_id_city_enc_total_item_cnt_day'
                 ]
    
    # extract the predictor columns which are not an element of index, target or exlusion columns
    pred_cols = [col for col in data_cols if col not in index_cols + tar_cols + excl_cols]
    
    # create a dictionary of the output columns
    model_cols_dict = {'index_cols':index_cols,
                       'tar_cols':tar_cols,
                       'excl_cols':excl_cols,
                       'pred_cols':pred_cols
                       }
    
    return model_cols_dict