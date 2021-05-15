# -*- coding: utf-8 -*-
"""
Created on Sat May 15 08:32:18 2021

@author: oislen
"""

import pandas as pd

def extract_feat_imp(cons, 
                     feat_imp, 
                     n = 20
                     ):
    
    """
    
    Extract Feature Importance Documentation
    
    Function Overview
    
    This function extracts the top n features from a given feature importance summary.
    
    Defaults
    
    extract_feat_imp(cons, 
                     feat_imp, 
                     n = 20
                     )
    
    Parameters
    
    cons - Python Module, the programme constants for the competition
    feat_imp - DataFrame, the results of the feature importance analysis
    n - Integer, the number of best features to extract from the feature importance analysis, default is 20
    
    Returns
    
    model_cols_dict - Dictionary, the extract n predictors based on the feature importance results, as well as the index columns, required columns and target columsn
    
    Example
    
    extract_feat_imp(cons = cons, 
                     feat_imp = 'randforest', 
                     n = 20
                     )
    
    """
    
    # input error handling
    # feat_imp
    valid_feat_imp = ['randforest', 'gradboost' ]
    if feat_imp not in valid_feat_imp:
        mess = 'Input Error: specified feat_imp parameter {} must be one of {}'.format(feat_imp, valid_feat_imp)
        raise ValueError(mess)
    # n features
    valid_n = type(n) == int and n > 0
    if  not (valid_n):
        mess = 'Input Error: specified n paramter {} must be a postive integer'.format(n)
        raise ValueError(mess)
    
    # set predefined index, target and required predictor columns
    index_cols = ['primary_key', 'ID', 'data_split', 'meta_level', 'holdout_subset_ind', 'no_sales_hist_ind']
    tar_cols = ['item_cnt_day']
    req_cols = ['year', 'month', 'date_block_num', 'item_id', 'shop_id']
    
    # load in feature importance file
    if feat_imp == 'randforest':
        feat_imp_df= pd.read_csv(cons.randforest_feat_imp)
    elif feat_imp == 'gradboost':
        feat_imp_df = pd.read_csv(cons.gradboost_feat_imp)
    
    # extract the required n features
    feat_imp_cols = feat_imp_df.loc[0:n, 'attr'].tolist()
    
    # return unique set of predictors
    pred_cols = pd.Series(feat_imp_cols + req_cols).drop_duplicates().tolist()
    
    # create a dictionary of the output columns
    model_cols_dict = {'index_cols':index_cols,
                       'req_cols':req_cols,
                       'tar_cols':tar_cols,
                       'pred_cols':pred_cols
                       }
    
    return model_cols_dict
