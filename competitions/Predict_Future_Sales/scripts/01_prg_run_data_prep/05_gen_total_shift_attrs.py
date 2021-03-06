# -*- coding: utf-8 -*-
"""
Created on Thu May 14 18:36:43 2020

@author: oislen
"""

import pandas as pd
from reference.gen_attr_agg_totals import gen_attr_agg_totals
from reference.months_since_purchase import months_since_purchase
from reference.n_price_changes import n_price_changes
from reference.recast_df import recast_df

def gen_shift_attrs(cons):
    
    """
    
    Generate Shift Attributes Documentation
    
    Function Overview
    
    This function generates shift attributes for a specified column.
    It is achieved by creating a pivot table on shop_id and item_id by date_block_num.
    Additiona; attributes are also created such as months since first and last purchase, and number of item price changes.
    
    Defaults
    
    gen_shift_attrs(cons)
    
    Parameters
    
    cons - Python Module, the programme constants for the competition
    
    Returns
    
    0 for successful execution
    
    Example
    
    gen_shift_attrs(cons = cons)
    
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
    base_agg_comp = gen_attr_agg_totals(dataset = base_agg_comp,
                                        values = values_total,
                                        index = index_total,
                                        columns = ['shop_id'],
                                        fill_value = fill_na
                                        )

    print('Calculating sold item totals for item id ...')
    
 
    # generate item sell totals
    base_agg_comp = gen_attr_agg_totals(dataset = base_agg_comp,
                                        values = values_total,
                                        index = index_total,
                                        columns = ['item_id'],
                                        fill_value = fill_na
                                        )
    
    print('Calculating sold item totals for item category id ...')
    
 
    # generate item sell totals
    base_agg_comp = gen_attr_agg_totals(dataset = base_agg_comp,
                                        values = values_total,
                                        index = index_total,
                                        columns = ['item_category_id'],
                                        fill_value = fill_na
                                        )
    
    print('Calculating sold item totals for item category id and shop id ...')
    
 
    # generate item sell totals
    base_agg_comp = gen_attr_agg_totals(dataset = base_agg_comp,
                                        values = values_total,
                                        index = index_total,
                                        columns = ['shop_id', 'item_category_id'],
                                        fill_value = fill_na
                                        )
    
    print('Calculating sold item totals for city ...')
    
 
    # generate item sell totals
    base_agg_comp = gen_attr_agg_totals(dataset = base_agg_comp,
                                        values = values_total,
                                        index = index_total,
                                        columns = ['city_enc'],
                                        fill_value = fill_na
                                        )
    

    print('Calculating sold item totals for item id and city ...')
    
 
    # generate item sell totals
    base_agg_comp = gen_attr_agg_totals(dataset = base_agg_comp,
                                        values = values_total,
                                        index = index_total,
                                        columns = ['item_id', 'city_enc'],
                                        fill_value = fill_na
                                        )
    
    
    print('Generating months since first and last purchases ...')
    
    print('shop_id & item_id ...')
    
    # generate item sell totals
    base_agg_comp = months_since_purchase(dataset = base_agg_comp,
                                          values =  ['item_cnt_day'],
                                          index = ['date_block_num'],
                                          columns = ['shop_id', 'item_id']
                                          )
    
    print('item_id ...')
    
    # generate item sell totals
    base_agg_comp = months_since_purchase(dataset = base_agg_comp,
                                          values =  ['item_cnt_day'],
                                          index = ['date_block_num'],
                                          columns = ['item_id']
                                          )
    
    print('Generating number of price changes ...')
    
    base_agg_comp = n_price_changes(dataset = base_agg_comp,
                                    values = ['item_price'],
                                    index = ['date_block_num'],
                                    columns = ['shop_id', 'item_id']
                                    )

    print('Recast data ...')
    
    base_agg_comp = recast_df(dataset = base_agg_comp)
    
    # output file to feather file
    model_data = base_agg_comp.reset_index(drop = True)
    model_data.to_feather(cons.base_agg_shft_fpath)
    
    return 0
    