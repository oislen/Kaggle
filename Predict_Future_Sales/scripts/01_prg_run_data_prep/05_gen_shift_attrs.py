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
    
    # load in the raw data
    item_categories, items, sales_train, sample_submission, shops, test = utl.load_files('clean', cons)
    
    # output aggreated base data as feather file
    base_agg_comp = pd.read_feather(cons.base_agg_comp_fpath)

    # set shift function inputs
    values = ['item_cnt_day']
    index = ['date_block_num']
    columns = ['shop_id', 'item_id']
    n_shifts = 24
    
    print('Running shift attributes ...')
    
    # create shift attributes
    shift_attrs = utl.gen_shift_attr(dataset = base_agg_comp, 
                                     values = values, 
                                     index = index, 
                                     columns = columns,
                                     n_shifts = n_shifts
                                     )
    
    print('Consolidating all attributes ...')
    
    # set the join columns
    join_cols = columns + index
    
    # join shift attributes back to base data
    data_join = pd.merge(left = base_agg_comp,
                         right = shift_attrs,
                         on = join_cols,
                         how = 'left'
                         )
    
    print('Outputting results ...')
    
    # output file to feather file
    data_join.to_feather(cons.base_agg_shft_fpath)
    
    return
    