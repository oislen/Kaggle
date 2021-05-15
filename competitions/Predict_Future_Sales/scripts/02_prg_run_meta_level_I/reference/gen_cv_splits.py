# -*- coding: utf-8 -*-
"""
Created on Sat May 15 08:32:17 2021

@author: oislen
"""
   
def gen_cv_splits(dataset,
                  train_cv_split_dict
                  ):
    
    """
    
    Generate Cross-Validation Splits Documentation
    
    Function Overview
    
    This function splits up the modelling data into cross-validation splits given a dictionary of desired splits.
    
    Defaults
    
    gen_cv_splits(dataset,
                  train_cv_split_dict
                  )
    
    Parameters
    
    dataset - DataFrame, the modelling data to split into cross-validation splits
    train_cv_split_dict - Dictionary, the desired splits to use when generating the cross-validation splits
    
    Returns
    
    cv_list - List of dictionarys, the resulting cross-validation data splits of the modelling data
    
    Example
    
    gen_cv_splits(dataset = base,
                  train_cv_split_dict =  [{'train_sub':28, 'valid_sub':29}]
                  )
    
    """
    
    # take a deep copy of the data
    data = dataset.copy(True)
    
    # create an empty list to append the cv splits to
    cv_list = []
    
    # for each split in the specified cv data splits dictionary
    for splits_limits in train_cv_split_dict:
        
        print(splits_limits)
        print('taking subset of data ...')
        
        # subset the required columns for performing the data split
        sub_cols = ['date_block_num', 'data_split', 'primary_key']
        data_sub = data[sub_cols]
        
        print('creating data split sub column ...')
        
        # extract out the train, valid and test subset limits
        trn_sl = splits_limits['train_sub']
        vld_sl = splits_limits['valid_sub']
        
        print('extracting data splits ...')
    
        # define data split filters
        filt_train = data_sub['date_block_num'] <= trn_sl
        filt_valid = data_sub['date_block_num'] == vld_sl
            
        # extract out the data splits
        train_data = data_sub[filt_train].index.values.astype(int)
        valid_data = data_sub[filt_valid].index.values.astype(int)
    
        print('creating output dictionary ...')
    
        # create a dictionary of data splits
        data_splits_dict = {'train_data_idx':train_data,
                            'valid_data_idx':valid_data
                            }
        
        # append the cv splits to the output list
        cv_list.append((data_splits_dict['train_data_idx'], data_splits_dict['valid_data_idx']))
    
    return cv_list
