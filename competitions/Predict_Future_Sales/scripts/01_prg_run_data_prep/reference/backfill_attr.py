# -*- coding: utf-8 -*-
"""
Created on Mon May 10 07:55:50 2021

@author: oislen
"""

import pandas as pd
import numpy as np

def backfill_attr(dataset, 
                  pivot_values, 
                  fillna = None,
                  pivot_index = ['year', 'month'], 
                  pivot_columns = ['shop_id', 'item_id'], 
                  ffill = False
                  ):
    
    """
    
    Back Fill Attribute Documentation
    
    Function Overview
    
    This function backfills attributes by propagating the last valid observation
    
    Defaults
    
    backfill_attr(dataset, 
                  pivot_values, 
                  fillna = None,
                  pivot_index = ['year', 'month'], 
                  pivot_columns = ['shop_id', 'item_id'], 
                  ffill = False
                  )
    
    Parameters
    
    dataset - DataFrame, the data to back fill
    pivot_values - List of Strings, the attribute values to back fill
    fillna - Numeric, default value for any previously unknown values
    pivot_index - List of Strings, the index to use for the initial pivot table
    pivot_columns - List of Strings, the columns to use for the initial pivot table
    ffill - Boolean, wheather to perform the back propagation
    
    Returns
    
    unstack - DataFrame, the back filled attribute
    
    Example
    
    utl.backfill_attr(dataset = agg_base, 
                      pivot_values = ['item_price'], 
                      fillna = -999,
                      pivot_index = ['year', 'month'], 
                      pivot_columns = ['shop_id', 'item_id'], 
                      ffill = True
                      )
    
    """
        
    print('Back filling {} ...'.format(pivot_values[0]))
    
    # set up for finding price of an item
    table = pd.pivot_table(data = dataset, 
                           values = pivot_values,
                           index = pivot_index,
                           columns = pivot_columns,
                           aggfunc = np.sum,
                           fill_value = None,
                           dropna = True
                           )
    
    # if forward fill
    if ffill:
            
        # want to forward fill from given price to next price
        table = table.ffill(axis = 0)
        
    if fillna != None:
            
        # add in unknown default
        table = table.fillna(fillna)
        
    # double unstack to get data by year, month, item_id and shop_id
    unstack = table.stack().stack().reset_index()

    return unstack
