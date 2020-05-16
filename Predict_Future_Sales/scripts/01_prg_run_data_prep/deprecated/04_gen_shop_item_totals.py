# -*- coding: utf-8 -*-
"""
Created on Sat May 16 15:49:13 2020

@author: oislen
"""

import pandas as pd
import reference.utilities as utl

def gen_shop_item_totals(cons):
    
    """
    
    Generate Shop / Item Totals Documentation
    
    Function Overview
    
    This funciton generates the total shop and item sales.
    
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
    
    # output as feather file
    item_total_data.to_feather(cons.base_agg_totl_fpath)
    
    return
    