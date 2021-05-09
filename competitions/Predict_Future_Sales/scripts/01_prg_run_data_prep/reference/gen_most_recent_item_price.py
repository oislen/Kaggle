# -*- coding: utf-8 -*-
"""
Created on Sun May  9 15:47:30 2021

@author: oislen
"""

def gen_most_recent_item_price(dataset):
    
    """
    
    Generate Most Recent Item Price Documentation
    
    Function Overview
    
    This function finds the most recent item price for a given dataset by:
        * sorting according to ['date_block_num', 'shop_id', 'item_id'] 
        * performing a group by using ['item_id']
        * aggregating with{'item_price':'last'}
    
    Defaults
    
    gen_most_recent_item_price(dataset)
    
    Parameters
    
    dataset - DataFrame, the data to find the most recent item prices for
    
    Returns
    
    recent_price - DataFrame, the most recent item prices
    
    Example
    
    gen_most_recent_item_price(dataset = clean_data)
    
    """
    
    # Note: not all shops sell every item
    # Note: not all shops are the same price
    
    # define the columns to sort by
    sort_cols = ['date_block_num', 'shop_id', 'item_id']
    
    # define the column to group by
    group_cols = ['item_id']
    
    # define the aggreation dictionary
    agg_dict = {'item_price':'last'}
    
    # apply the sorting
    agg_base_sort = dataset.sort_values(by = sort_cols)
    
    # apply the group by and aggregation
    recent_price = agg_base_sort.groupby(group_cols, as_index = False).agg(agg_dict)
    
    return recent_price
    