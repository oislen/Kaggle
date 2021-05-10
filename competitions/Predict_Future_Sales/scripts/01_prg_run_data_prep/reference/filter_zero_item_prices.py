# -*- coding: utf-8 -*-
"""
Created on Mon May 10 09:12:09 2021

@author: oislen
"""

import pandas as pd

def filter_zero_item_prices(dataset):
    
    """
    
    Filter Zero Item Prices Documentation
    
    Function Overview
    
    This function removes item prices with zero recorded sales.
    
    Defaults
    
    filter_zero_item_prices(dataset)
    
    Parameters
    
    dataset - DataFrame, the data to remove item prices with zero sales from
    
    Returns
    
    filt_data - DataFrame, the filtered item price data
    
    """
    
    # load in pickled holdout item shop id combination
    #holdout_shop_item_id_comb = pk.load(open(cons.holdout_shop_item_id_comb, 'rb'))
    dataset['shop_item_id'].nunique()
    
    # create filters for item price and holdout shop item combination
    price_tab = pd.pivot_table(data = dataset,
                               index = 'date_block_num',
                               columns = ['shop_id', 'item_id'],
                               values = 'item_price'
                               )
    all_zero_price = (price_tab == 0).all()
    keep_shop_item_comb = all_zero_price[all_zero_price].reset_index().drop(columns = 0)
    keep_shop_item_comb_series = keep_shop_item_comb['shop_id'].astype(str) + '_' + keep_shop_item_comb['item_id'].astype(str)
    
    filt_shop_item_id = dataset['shop_item_id'].isin(keep_shop_item_comb_series)
    filt_default_price = dataset['item_price'] == 0
    
    filt_data = dataset[~filt_default_price | filt_shop_item_id]
    
    return filt_data