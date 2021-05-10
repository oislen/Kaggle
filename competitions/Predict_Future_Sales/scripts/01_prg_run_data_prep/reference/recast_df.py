# -*- coding: utf-8 -*-
"""
Created on Mon May 10 08:59:38 2021

@author: oislen
"""

import numpy as np
import reference.extract_float_int_cols as extract_float_int_cols

def recast_df(dataset, 
              sample_size = 100000
              ):
    
    """
    
    Recast DataFrame Documentation
    
    Function Overview
    
    This function recasts the various columns of a given dataset to a relevant data type of lower bits in order to save memory and compute space
    
    Defaults
    
    recast_df(dataset, 
              sample_size = 100000
              )
    
    Parameters
    
    dataset - DataFrame, the data to recast to lower bit types
    sample_size - Integer, the sample of the data to take when determining optimal bit size
    
    Returns
    
    data - DataFrame, the recast data
    
    Example
    
    recast_df(dataset = model_data, 
              sample_size = 100000
              )
        
    """
    
    print('Copying data ...')
    
    data = dataset.copy(True)
    n_rows = data.shape[0]
    
    print('Converting non-null columns to int32s ...')
    
    # check missing values
    n_null = data.isnull().sum()
    nonull_cols = n_null[n_null == 0].index 
    
    # extract int and float columns
    int_cols, float_cols = extract_float_int_cols(data = data)
    
    # find non-null floats
    nonull_float_cols = [col for col in nonull_cols if col in float_cols]
    
    if nonull_float_cols != []:
        
        # check a random sample 1000 records for all '.0' decimals
        n_sample = [n_rows if n_rows < sample_size else sample_size][0]
        data_str = data[nonull_float_cols].sample(n_sample, random_state = 1234).astype(str)
        true_floats = data_str.apply(lambda x: x.str.contains('\.0').all(), axis = 0)
        
        # cast true floats to int
        cast_to_int_cols = true_floats[true_floats].index
        data[cast_to_int_cols] = data[cast_to_int_cols].astype(np.int32)
          
        # extract int and float columns
        int_cols, float_cols = extract_float_int_cols(data = data)
        
    print('Generating column min / max values ...')
    
    # get maximum values
    max_int = data[int_cols].max().rename('max').reset_index()
    min_int = data[int_cols].min().rename('min').reset_index()
    max_float = data[float_cols].max().rename('max').reset_index()
    min_float = data[float_cols].min().rename('min').reset_index()
    
    # join min max values
    join_int = min_int.merge(max_int, on = 'index', how = 'inner')
    join_float = min_float.merge(max_float, on = 'index', how = 'inner')
    
    # create dtype columns
    join_int['dtype'] = np.nan
    join_float['dtype'] = np.nan
    
    print('Defining dtype ranges for recasting ...')
    
    # get the data type limits
    lim_int8 = np.iinfo(np.int8)
    lim_int16 = np.iinfo(np.int16)
    lim_int32 = np.iinfo(np.int32)
    lim_int64 = np.iinfo(np.int64)
    #lim_float16 = np.finfo(np.float16)
    lim_float32 = np.finfo(np.float32)
    lim_float64 = np.finfo(np.float64)
    
    # get data type ranges
    range_int8 = (lim_int8.min, lim_int8.max) 
    range_int16 = (lim_int16.min, lim_int16.max) 
    range_int32 = (lim_int32.min, lim_int32.max) 
    range_int64 = (lim_int64.min, lim_int64.max) 
    #range_float16 = (lim_float16.min, lim_float16.max) 
    range_float32 = (lim_float32.min, lim_float32.max) 
    range_float64 = (lim_float64.min, lim_float64.max)  
    
    # recast int data
    if join_int.empty == False:
        
        print('Finding optimal int data type cast ...')
        
        # get apply min / max search
        join_int['dtype'] = join_int.apply(lambda x: 'int64' if x['min'] >= range_int64[0] and x['max'] <= range_int64[1] else x['dtype'], axis = 1)
        join_int['dtype'] = join_int.apply(lambda x: 'int32' if x['min'] >= range_int32[0] and x['max'] <= range_int32[1] else x['dtype'], axis = 1)
        join_int['dtype'] = join_int.apply(lambda x: 'int16' if x['min'] >= range_int16[0] and x['max'] <= range_int16[1] else x['dtype'], axis = 1)
        join_int['dtype'] = join_int.apply(lambda x: 'int8' if x['min'] >= range_int8[0] and x['max'] <= range_int8[1] else x['dtype'], axis = 1)
        
        print('Recasting data ...')
        
        # extract out the relevant data types
        filt_cast_int64 = join_int['dtype'] == 'int64'
        filt_cast_int32 = join_int['dtype'] == 'int32'
        filt_cast_int16 = join_int['dtype'] == 'int16'
        filt_cast_int8 = join_int['dtype'] == 'int8'
        cast_cols_int64 = join_int.loc[filt_cast_int64, 'index']
        cast_cols_int32 = join_int.loc[filt_cast_int32, 'index']
        cast_cols_int16 = join_int.loc[filt_cast_int16, 'index']
        cast_cols_int8 = join_int.loc[filt_cast_int8, 'index']
        
        # recast the data
        data[cast_cols_int64] = data[cast_cols_int64].astype(np.int64)
        data[cast_cols_int32] = data[cast_cols_int32].astype(np.int32)
        data[cast_cols_int16] = data[cast_cols_int16].astype(np.int16)
        data[cast_cols_int8] = data[cast_cols_int8].astype(np.int8)
        
    # recast float data
    if join_float.empty == False:
    
        print('Finding optimal float data type cast ...')
        
        # get apply min / max search
        join_float['dtype'] = join_float.apply(lambda x: 'float64' if x['min'] >= range_float64[0] and x['max'] <= range_float64[1] else x['dtype'], axis = 1)
        join_float['dtype'] = join_float.apply(lambda x: 'float32' if x['min'] >= range_float32[0] and x['max'] <= range_float32[1] else x['dtype'], axis = 1)
        #join_float['dtype'] = join_float.apply(lambda x: 'float16' if x['min'] >= range_float16[0] and x['max'] <= range_float16[1] else x['dtype'], axis = 1)
        
        print('Recasting data ...')
        
        # extract out the relevant data types
        filt_cast_float64 = join_float['dtype'] == 'float64'
        filt_cast_float32 = join_float['dtype'] == 'float32'
        cast_cols_float64 = join_float.loc[filt_cast_float64, 'index']
        cast_cols_float32 = join_float.loc[filt_cast_float32, 'index']
        
        # recast the data
        data[cast_cols_float64] = data[cast_cols_float64].astype(np.float64)
        data[cast_cols_float32] = data[cast_cols_float32].astype(np.float32)
        
    print(data.dtypes.value_counts())
        
    return data
