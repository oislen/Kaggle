# -*- coding: utf-8 -*-
"""
Created on Thu May 14 18:36:43 2020

@author: oislen
"""

import pandas as pd
import numpy as np
import reference.clean_utilities as utl

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
    
    shape = base_agg_comp.shape
    
    print(shape)
    
    # set function inputs for total aggregate attributes
    values_total = ['item_cnt_day']
    index_total = ['date_block_num']
    fill_na = 0
    
    # set function inputs for item count shift attributes
    index_shift = ['date_block_num']
    columns_shift = ['shop_id', 'item_id']
    lags = [1, 2, 3, 4, 6, 12]
    
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
    
    #-- Lag Item Cnt Shifts --#
    
    # create shift attributes
    base_agg_comp = utl.gen_shift_attr(dataset = base_agg_comp, 
                                       values = ['item_cnt_day'], 
                                       index = index_shift, 
                                       columns = columns_shift,
                                       lags = lags,
                                       fill_value = fill_na
                                       )
    
    #-- Lag Shop Total Shifts --#
    
    # create shift attributes
    base_agg_comp = utl.gen_shift_attr(dataset = base_agg_comp, 
                                       values = ['shop_id_total_item_cnt_day'], 
                                       index = index_shift, 
                                       columns = columns_shift,
                                       lags = lags,
                                       fill_value = fill_na
                                       )
    
    #--Lag Item Total Shifts --#
    
    # create shift attributes
    base_agg_comp = utl.gen_shift_attr(dataset = base_agg_comp, 
                                       values = ['item_id_total_item_cnt_day'], 
                                       index = index_shift, 
                                       columns = columns_shift,
                                       lags = lags,
                                       fill_value = fill_na
                                       )
     
    #-- Lag Item Price --#
    
    # create shift attributes
    base_agg_comp = utl.gen_shift_attr(dataset = base_agg_comp, 
                                       values = ['item_price'], 
                                       index = index_shift, 
                                       columns = columns_shift,
                                       lags = [1],
                                       fill_value = fill_na
                                       )
    #print('Replace -999s with missing values ...')
    
    #base_agg_comp = base_agg_comp.replace(-999, np.nan)
    
    print('Removing 1st year of data due to lagged attributes ...')
    
    filt_1st_year = base_agg_comp['date_block_num'] >= 12
    base_agg_comp = base_agg_comp[filt_1st_year]
    shape = base_agg_comp.shape
    
    print('Outputting results {} ...'.format(shape))
    
    # output file to feather file
    base_agg_comp = base_agg_comp.reset_index(drop = True)
    base_agg_comp.to_feather(cons.base_agg_shft_fpath)
    
    return
    