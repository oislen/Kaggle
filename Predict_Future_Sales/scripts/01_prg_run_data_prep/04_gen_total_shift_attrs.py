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
    
    # set function inputs for total aggregate attributes
    values_total = ['item_cnt_day']
    index_total = ['date_block_num']
    fill_na = 0
    
    # set function inputs for item count shift attributes
    index_shift = ['date_block_num']
    columns_shift = ['shop_id', 'item_id']
    n_shifts = 24
    
    # set additional function inputs for total shift attributes
    #columns_shift_shop_total = ['shop_id']
    #columns_shift_item_total = ['item_id']
    
    print('Calculating sold item totals for shop id ...')
 
    # generate the shop sell totals
    base_agg_comp = utl.gen_attr_agg_totals(dataset = base_agg_comp,
                                            values = values_total,
                                            index = index_total,
                                            columns = ['shop_id'],
                                            fill_value = fill_na
                                            )

    print('Calculating sold item totals for item id ...')
    
 
    # generate item sell totals
    base_agg_comp = utl.gen_attr_agg_totals(dataset = base_agg_comp,
                                            values = values_total,
                                            index = index_total,
                                            columns = ['item_id'],
                                            fill_value = fill_na
                                            )
    

    print('Running shift attributes for item cnt ...')
    
    # create shift attributes
    base_agg_comp = utl.gen_shift_attr(dataset = base_agg_comp, 
                                       values = ['item_cnt_day'], 
                                       index = index_shift, 
                                       columns = columns_shift,
                                       n_shifts = n_shifts,
                                       fill_value = fill_na
                                       )
    
    # create shift attributes
    base_agg_comp = utl.gen_shift_attr(dataset = base_agg_comp, 
                                       values = ['shop_id_total_item_cnt_day'], 
                                       index = index_shift, 
                                       columns = columns_shift,
                                       n_shifts = n_shifts,
                                       fill_value = fill_na
                                       )
    
    # create shift attributes
    base_agg_comp = utl.gen_shift_attr(dataset = base_agg_comp, 
                                       values = ['item_id_total_item_cnt_day'], 
                                       index = index_shift, 
                                       columns = columns_shift,
                                       n_shifts = n_shifts,
                                       fill_value = fill_na
                                       )
    
    print('Outputting results ...')
    
    # output file to feather file
    base_agg_comp.to_feather(cons.base_agg_shft_fpath)
    
    return
    