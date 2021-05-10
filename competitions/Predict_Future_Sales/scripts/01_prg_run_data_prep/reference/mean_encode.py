# -*- coding: utf-8 -*-
"""
Created on Mon May 10 08:44:15 2021

@author: oislen
"""

def mean_encode(dataset, 
                attr, 
                tar, 
                alpha = 100,
                encode_type = 'leave-one-out'
                ):
    
    """
    
    Mean Encode Documentation
    
    Function Overview
    
    This function generates mean encoded attributes for a given dataset
    
    Defaults
    
    mean_encode(dataset, 
                attr, 
                tar, 
                alpha = 100,
                encode_type = 'leave-one-out'
                )
    
    Parameters
    
    dataset - DataFrame, the data to generate mean encoded attributes from
    attr - String, the column to group by for the mean encoded attribute
    tar - String, the column values to use for the mean encoded attribute
    alpha - Numeric, the alpha regularisation factor to correct the mean encoded attribute with, default is 100
    encode_type - String, the type of mean encoding to perform
    
    Returns
    
    stat_enc - DataFrame, the mean encoded attribute
    
    Example
    
    mean_encodemean_encode(dataset = base_agg_comp, 
                           attr = ['date_block_num'], 
                           tar = 'item_cnt_day',
                           alpha = 100,
                           encode_type = 'leave-one-out'
                           )
       
    """
    
    # filter zeros
    #filt_zero_tar = dataset[tar] > 0
    filt_data = dataset.copy(True)#[filt_zero_tar]
    
    stat = filt_data[tar].mean()
    attr_name = '_'.join(attr)
    feat_name = '{}_mean_enc'.format(attr_name)
    
    if encode_type == 'leave-one-out':
        
        # mean encode
        target_sum  = filt_data.groupby(attr)[tar].transform('sum')
        n_objects = filt_data.groupby(attr)[tar].transform('count')
        stat_enc = (target_sum - filt_data[tar]) / (n_objects - 1)
        stat_enc = stat_enc.fillna(stat).rename(feat_name)
 
    elif encode_type == 'smoothing':
        
        # mean encode
        target_mean = filt_data.groupby(attr)[tar].transform('mean')
        n_objects = filt_data.groupby(attr)[tar].transform('count')
        stat_enc = (target_mean * n_objects + stat * alpha) / (n_objects + alpha)
        stat_enc = stat_enc.fillna(stat).rename(feat_name)
            
    elif encode_type == 'expanding_mean':
        
        # mean encode
        cumsum = filt_data.groupby(by = attr)[tar].cumsum() - filt_data[tar]
        cumcnt = filt_data.groupby(attr).cumcount()
        stat_enc = cumsum / cumcnt
        stat_enc = stat_enc.fillna(stat).rename(feat_name)
      
    return stat_enc
