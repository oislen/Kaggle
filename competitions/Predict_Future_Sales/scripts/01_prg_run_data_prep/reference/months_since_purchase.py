# -*- coding: utf-8 -*-
"""
Created on Mon May 10 08:33:01 2021

@author: oislen
"""

import pandas as pd
import numpy as np

def months_since_purchase(dataset, 
                          values, 
                          index, 
                          columns
                          ):
    
    """
    
    Months Since Purchase Documentation
    
    Funciton Overview
    
    This function generates the months since first and last purchase for a given index and column combination and attribute values
    
    Defaults
    
    months_since_purchase(dataset, 
                          values, 
                          index, 
                          columns
                          )
    
    Parameters
    
    dataset - DataFrame, the data to generate the months since purchase attributes from
    values - List of Strings, the column to use as the attribute values
    index - List of Strings, the index to use for the initial pivot table
    columns - List of Strings, the columns to use for the initial pivot table
    
    Returns
    
    out_data - DataFrame, the months since first and last purchase attributes
    
    Example
    
    gen_attr_agg_totals(dataset = base_agg_comp,
                        values = ['item_cnt_day'],
                        index = ['date_block_num'],
                        columns = ['item_id', 'city_enc'],
                        fill_value = 0
                        )
    """
    
    # take deep copy of data
    data = dataset.copy(True)
    
    # determine dimensions of data
    shape = data.shape
    print(shape)
    
    # create the attribute name
    attr = '_'.join(columns)
    
    # set the columns to re-join the data on
    join_cols = index + columns
    
    print('Creating pivot table ...')
    
    # create pivot table for item sale by date block
    sales_table = pd.pivot_table(data = data, 
                                 values = values,
                                 index = index,
                                 columns = columns,
                                 fill_value = 0,
                                 aggfunc = np.sum,
                                 dropna = True
                                 )
    
    print('Formatting table ...')
    
    # convert all postive cases to one and all zeros to missing
    filt_sales = sales_table >= 1
    sales_table[filt_sales] = 1
    tab = sales_table[filt_sales]
    
    print('Calculating months since first and last purchase ...')
    
    # calculate months since first purchase
    msfp = tab.ffill().notnull().cumsum()
    mslp = tab.bfill().isnull().cumsum()
    
    msfp_unstack = msfp.unstack().reset_index().drop(columns = ['level_0']).rename(columns = {0:'{}_months_first_rec'.format(attr)})
    mslp_unstack = mslp.unstack().reset_index().drop(columns = ['level_0']).rename(columns = {0:'{}_months_last_rec'.format(attr)})
    
    print('Joining back results ...')
    
    # join together the months since purchase attributes 
    msp_unstack = pd.merge(left = msfp_unstack, right = mslp_unstack, how = 'inner', on = join_cols)
    
    # join back to data
    out_data = data.merge(msp_unstack, how = 'left', on = join_cols)
    
    shape = out_data.shape
    print(shape)
    
    return out_data
