# -*- coding: utf-8 -*-
"""
Created on Sun May  9 15:37:43 2021

@author: oislen
"""

import pandas as pd
import numpy as np

def gen_shift_attr(dataset, 
                   values, 
                   index, 
                   columns, 
                   lags = 12,
                   fill_value = 0
                   ):
    
    """
    
    Generate Shift Attributes Documentation
    
    Function Overview
    
    This function generates shift attributes by pivoting the data using a given combination of columns and creating lags / shifts in the results.
    For instance the previous months sales for all shops can be lagged.
    
    Defaults
    
    gen_shift_attr(dataset, 
                   values, 
                   index, 
                   columns, 
                   lags = 12,
                   fill_value = 0
                   )
    
    Parameters
    
    dataset - DataFrame, the data to create shift attribtues from
    values - List of Strings, the values to use for the shift attributes
    index - List of Strings, the index for the initial pivot table
    columns - List of Strings, the columns for the initial pivot table
    lags - List of Integers, the various lag terms to use when creating the shift attributes
    fill_value - Numeric, the value to use for filling in missing values
    
    Returns
    
    data_out - DataFrame, the shift attributes
    
    Example
    
    gen_shift_attr(dataset = clean_data, 
                   values = ['item_cnt_day'],
                   index = ['date_block_num'],
                   columns = ['shop_id', 'item_id'], 
                   lags = 12,
                   fill_value = 0
                   )
    
    
    """
    
    # extract the relevant input info
    feat_name = values[0]
    join_cols = columns + index
    data_join = dataset[join_cols].copy(True)
    
    print('creating pivot table ...')
    
    # create pivot table for item sale by date block
    sales_table = pd.pivot_table(data = dataset, 
                                 values = values,
                                 index = index,
                                 columns = columns,
                                 fill_value = fill_value,
                                 aggfunc = np.sum,
                                 dropna = True
                                 )
    
    shape = sales_table.shape
    
    print(shape)
    
    # for each required shift
    for i in lags:
        
        print('Working on shift {} ...'.format(i))
        print('create shifts ...')
        
        # shift the data by i
        shift_data = sales_table.shift(i, fill_value = 0)
        shape = shift_data.shape     
        print(shape)
        
        print('unstacking data ...')
        
        # unstack the shifted attribute
        attr_name = '{}_shift_{}'.format(feat_name, i)
        unstack_data = shift_data.unstack()
        attr_shift_data = unstack_data.reset_index().drop(columns = ['level_0']).rename(columns = {0:attr_name})
        shape = attr_shift_data.shape
        print(shape)
        
        print('joining back results ...')
        
        # store the shifted attribute
        data_join = data_join.merge(attr_shift_data, on = join_cols, how = 'left')
        shape = data_join.shape
        print(shape)
        
    print('Consolidating all attributes ...')
    
    # set the join columns
    join_cols = columns + index
    
    # join shift attributes back to base data
    data_out = pd.merge(left = dataset,
                        right = data_join,
                        on = join_cols,
                        how = 'left'
                        )
    shape = data_out.shape
    print(shape)
    
    return data_out