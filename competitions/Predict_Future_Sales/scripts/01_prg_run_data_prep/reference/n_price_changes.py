# -*- coding: utf-8 -*-
"""
Created on Mon May 10 09:04:53 2021

@author: oislen
"""

import pandas as pd
import numpy as np

def n_price_changes(dataset, 
                    values, 
                    index, 
                    columns
                    ):
    
    """
    
    N Price Changes Documentation
    
    Function Overview
    
    This function generates the number of price changes attributes for a given combination of columns and attribute values
    
    Defaults
    
    n_price_changes(dataset, 
                    values, 
                    index, 
                    columns
                    )
    
    Parameters
    
    dataset - DataFrame, the data to generate the number of price changes attributes for
    values - List of Strings, the attribute values to use for the aggregation
    index - List of Strings, the index of the initial pivot table
    columns - List of Strings, the columns of the initial pivot table
    
    Returns
    
    data_out - DataFrame, the number of price changes attributes
    
    Example
    
    gen_attr_agg_totals(dataset = base_agg_comp,
                        values = ['item_cnt_day'],
                        index = ['date_block_num'],
                        columns = ['shop_id'],
                        fill_value = fill_na
                        )
    
    """
    
    data = dataset.copy(True)
    
    print('Calculating number of unique item prices ...')
    table = pd.pivot_table(data = dataset,
                           values = values,
                           index = index,
                           columns = columns,
                           aggfunc = np.sum,
                           fill_value = 0,
                           dropna = True
                           )
    
    print('Number of price changes ...')
    sales = table.apply(lambda x: ~x.duplicated(), axis = 0)
    sales_int = sales.astype(int)
    price_change = sales_int.cumsum() - 1
    
    print('unstacking data ...')
    price_unstack = price_change.unstack().reset_index().drop(columns = ['level_0']).rename(columns = {0:'n_price_changes'})
    
    join_cols = columns + index
    data_out = data.merge(price_unstack, on = join_cols, how = 'left')
    
    return data_out