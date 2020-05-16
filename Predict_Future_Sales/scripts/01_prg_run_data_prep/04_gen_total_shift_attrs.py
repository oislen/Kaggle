# -*- coding: utf-8 -*-
"""
Created on Thu May 14 18:36:43 2020

@author: oislen
"""

import pandas as pd
import reference.utilities as utl

def gen_shift_attrs(cons):
    
    """
    
    Generate Shift Attributes Documentation
    
    Function Overview
    
    This function generates shift attributes for a specified column.
    It is achieved by creating a pivot table on shop_id and item_id by date_block_num
    
    Defaults
    
    gen_shift_attrs(cons)
    
    Parameters
    
    cons - 
    
    Returns
    
    Outputs
    
    Example
    
    
    """
    
    # output aggreated base data as feather file
    base_agg_comp = pd.read_feather(cons.base_agg_comp_fpath)
    
    # set function inputs
    values = ['item_cnt_day']
    index = ['date_block_num']
    
    print('Calculating sold item totals for shop id ...')
 
    # generate the shop sell totals
    shop_total_data = utl.gen_attr_agg_totals(dataset = base_agg_comp,
                                              values = values,
                                              index = index,
                                              columns = ['shop_id'],
                                              fill_value = 0
                                              )

    print('Calculating sold item totals for item id ...')
    
 
    # generate item sell totals
    item_total_data = utl.gen_attr_agg_totals(dataset = shop_total_data,
                                              values = values,
                                              index = index,
                                              columns = ['item_id'],
                                              fill_value = 0
                                              )
    

    print('Running shift attributes for item cnt ...')
    
    # set shift function inputs
    values = ['item_cnt_day']
    index = ['date_block_num']
    columns = ['shop_id', 'item_id']
    n_shifts = 24
    
    # create shift attributes
    shift_attrs = utl.gen_shift_attr(dataset = item_total_data, 
                                     values = values, 
                                     index = index, 
                                     columns = columns,
                                     n_shifts = n_shifts
                                     )
    
    print('Outputting results ...')
    
    # output file to feather file
    shift_attrs.to_feather(cons.base_agg_shft_fpath)
    
    return
    