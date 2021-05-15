# -*- coding: utf-8 -*-
"""
Created on Sat May 15 08:32:18 2021

@author: oislen
"""

import pandas as pd

def extract_data_splits(dataset,
                        index_cols,
                        req_cols,
                        tar_cols,
                        pred_cols,
                        test_split_dict
                        ):
    
    """
    
    Extract Data Splits Documentation
    
    Function Overview
    
    This function splits the modelling data into training, validation, test, holdout and meta-level II sets.
    The training data is used to training the initial models.
    The valudation data is used to validate the initial trained models.
    The test data is used as an out of sample set to test the generalised performance of the initial trained models.
    The holdout data is used to make predictions from the initial models, which are then used to to train a meta level two stacked model.
    
    Defaults
    
    extract_data_splits(dataset,
                        index_cols,
                        req_cols,
                        tar_cols,
                        pred_cols,
                        test_split_dict
                        )
    
    Parameters
    
    dataset - DataFrame, the modelling data
    index_cols - List of Strings, the names of the index columns of the modelling data
    req_cols - List of Strings, the names of the required columns of the modelling data
    tar_cols - List of Strings, the names of the target columns of the modelling data
    pred_cols - List of Strings, the names of the predictor columns of the modelling data
    test_split_dict - Dictionary, the train, validation and test splitting configurations for the modelling data
    
    Returns
    
    data_splits_dict - Dictionary, the various data splits for modelling stored as DataFrames
    
    Example
    
    extract_data_splits(dataset = base,
                        index_cols = index_cols,
                        req_cols = req_cols,
                        tar_cols = tar_cols,
                        pred_cols = pred_cols,
                        test_split_dict = {'train_sub':29, 'valid_sub':32, 'test_sub':33}
                        )
    
    """
    
    # take a deep copy of the data
    data = dataset.copy(True)

    print('generating data splits ...')
    
    print('creating data split sub column ...')
    
    # extract out the train, valid and test subset limits
    trn_sl = test_split_dict['train_sub']
    vld_sl = test_split_dict['valid_sub']
    tst_sl = test_split_dict['test_sub']
    
    # error handling
    cons_holdout = (trn_sl != 34) and (vld_sl != 34) and (tst_sl != 34)
    if not cons_holdout:
        mess = 'Input Error: the specified holdout subset limit is duplicated in the train, validation or test subset limits'
        raise ValueError(mess)
    cons_sl = (trn_sl < vld_sl) and (vld_sl < tst_sl) and (tst_sl < 34)
    if not cons_sl:
        mess = 'Input Error: the specified train subset limits {}, validation subset limits {} and test subset limits {} are not time orientated.'.format(trn_sl, vld_sl, tst_sl)
        raise ValueError(mess)
        
    print('extracting data splits ...')
    
    # define data split filters
    filt_train = data['date_block_num'] <= trn_sl
    filt_valid = (data['date_block_num'] > trn_sl) & (data['date_block_num'] <= vld_sl)
    filt_test = (data['date_block_num'] > vld_sl) & (data['date_block_num'] <= tst_sl)
    filt_holdout = data['date_block_num'] == 34
    filt_meta_lvl_II = data['meta_level'].isin(['level_2', 'holdout'])
        
    # extract out the data splits
    train_data = data[filt_train]
    valid_data = data[filt_valid]
    test_data = data[filt_test]
    holdout_data = data[filt_holdout]
    meta_lvl_II = data[filt_meta_lvl_II]
    
    # define the X_cols and y_cols
    X_cols = index_cols + pred_cols
    y_cols = index_cols + req_cols + tar_cols
    
    print(X_cols)
    
    # check if duplicate values are occuring
    if (pd.Series(X_cols).value_counts() > 1).any() or (pd.Series(y_cols).value_counts() > 1).any():
        # return error message and fail
        raise ValueError('Duplicate columns are being passed to the extract data splitter.')
    
    # split datasets into train, valid, test and holdout
    X_train = train_data[X_cols]
    y_train = train_data[y_cols]
    X_valid = valid_data[X_cols]
    y_valid = valid_data[y_cols]
    X_test = test_data[X_cols]
    y_test = test_data[y_cols]
    X_holdout = holdout_data[X_cols]
    y_holdout = holdout_data[y_cols]
    X_meta_lvl_II = meta_lvl_II[X_cols]
    y_meta_lvl_II = meta_lvl_II[y_cols]
    
    print('creating output dictionary ...')
    
    # create a dictionary of data splits
    data_splits_dict = {'train_data':train_data,
                        'valid_data':valid_data,
                        'test_data':test_data,
                        'holdout_data':holdout_data,
                        'meta_lvl_II':meta_lvl_II,
                        'X_train':X_train,
                        'y_train':y_train,
                        'X_valid':X_valid,
                        'y_valid':y_valid,
                        'X_test':X_test,
                        'y_test':y_test,
                        'X_holdout':X_holdout,
                        'y_holdout':y_holdout,
                        'X_meta_lvl_II':X_meta_lvl_II,
                        'y_meta_lvl_II':y_meta_lvl_II
                        }

    return data_splits_dict

