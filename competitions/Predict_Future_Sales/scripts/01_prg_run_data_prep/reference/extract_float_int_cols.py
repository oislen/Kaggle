# -*- coding: utf-8 -*-
"""
Created on Mon May 10 08:55:05 2021

@author: oislen
"""

import numpy as np

def extract_float_int_cols(data):
    
    """
    
    Extract Float / Integer Columns Documentation
    
    Function Overview
    
    This function extracts out float and integer columns from a given dataset
    
    Defaults
    
    extract_float_int_cols(data)
    
    Parameters
    
    data - DataFrame, the data to extract float and integer columns from
    
    Returns
    
    int_cols - Pandas.Series, the integer columns of the given dataset
    float_cols - Pandas.Series, the float columns of the given dataset
    
    Example
    
    extract_float_int_cols(data = model_data)
    
    """
    
    data_cols = data.columns
    data_dtypes = data.dtypes
    
    print(data_dtypes.value_counts())
     
    # filter various int data types
    filt_int8 = data_dtypes ==  np.int8
    filt_int16 = data_dtypes ==  np.int16
    filt_int32 = data_dtypes ==  np.int32
    filt_int64 = data_dtypes ==  np.int64
    
    # filter various float data types
    filt_float16 = data_dtypes ==  np.float16
    filt_float32 = data_dtypes ==  np.float32
    filt_float64 = data_dtypes ==  np.float64
    
    # extract out the integer and float columns
    int_cols = data_cols[filt_int8 | filt_int16 | filt_int32 | filt_int64]
    float_cols = data_cols[filt_float16 | filt_float32 | filt_float64]
    
    return int_cols, float_cols