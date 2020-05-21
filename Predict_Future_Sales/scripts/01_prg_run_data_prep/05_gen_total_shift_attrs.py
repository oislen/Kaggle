# -*- coding: utf-8 -*-
"""
Created on Thu May 14 18:36:43 2020

@author: oislen
"""

import pandas as pd
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
    base_agg_comp = pd.read_feather(cons.base_agg_supp_fpath)
    
    shape = base_agg_comp.shape
    
    print(shape)
    
    # set function inputs for total aggregate attributes
    values_total = ['item_cnt_day']
    index_total = ['date_block_num']
    fill_na = 0
    
    # set additional function inputs for total shift attributes
    #columns_shift_shop_total = ['shop_id']
    #columns_shift_item_total = ['item_id']
    
    #TODO: aggregate total month sales and lag by one
    #TODO: aggregate by shop category and sub catgory, city
    
    ###############################
    #-- Mean / Total Aggregates --#
    ###############################
    
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
    
    print('Calculating sold item totals for item category id ...')
    
 
    # generate item sell totals
    base_agg_comp = utl.gen_attr_agg_totals(dataset = base_agg_comp,
                                            values = values_total,
                                            index = index_total,
                                            columns = ['item_category_id'],
                                            fill_value = fill_na
                                            )
    
    print('Calculating sold item totals for item category id and shop id ...')
    
 
    # generate item sell totals
    base_agg_comp = utl.gen_attr_agg_totals(dataset = base_agg_comp,
                                            values = values_total,
                                            index = index_total,
                                            columns = ['shop_id', 'item_category_id'],
                                            fill_value = fill_na
                                            )
    
    print('Calculating sold item totals for city ...')
    
 
    # generate item sell totals
    base_agg_comp = utl.gen_attr_agg_totals(dataset = base_agg_comp,
                                            values = values_total,
                                            index = index_total,
                                            columns = ['city_enc'],
                                            fill_value = fill_na
                                            )
    

    print('Calculating sold item totals for item id and city ...')
    
 
    # generate item sell totals
    base_agg_comp = utl.gen_attr_agg_totals(dataset = base_agg_comp,
                                            values = values_total,
                                            index = index_total,
                                            columns = ['item_id', 'city_enc'],
                                            fill_value = fill_na
                                            )
    
    
    print('Generating months since first and last purchases ...')
    
    print('shop_id & item_id ...')
    
    # generate item sell totals
    base_agg_comp = utl.months_since_purchase(dataset = base_agg_comp,
                                              values =  ['item_cnt_day'],
                                              index = ['date_block_num'],
                                              columns = ['shop_id', 'item_id']
                                              )
    
    print('item_id ...')
    
    # generate item sell totals
    base_agg_comp = utl.months_since_purchase(dataset = base_agg_comp,
                                              values =  ['item_cnt_day'],
                                              index = ['date_block_num'],
                                              columns = ['item_id']
                                              )
    
    
    

    
    # output file to feather file
    model_data = base_agg_comp.reset_index(drop = True)
    model_data.to_feather(cons.base_agg_shft_fpath)
    
    return
    