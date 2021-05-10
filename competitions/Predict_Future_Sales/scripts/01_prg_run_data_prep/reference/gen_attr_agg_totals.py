# -*- coding: utf-8 -*-
"""
Created on Mon May 10 08:18:33 2021

@author: oislen
"""

import pandas as pd
import numpy as np

def gen_attr_agg_totals(dataset, 
                        values, 
                        index, 
                        columns, 
                        aggfunc = np.mean,
                        fill_value = 0
                        ):
    
    """
    
    Generate Attribute Aggregation Totals Documentation
    
    Funciton Overview
    
    This function generates aggregate values for a given combination of index columns and attribute values
    
    Defaults
    
    gen_attr_agg_totals(dataset,
                        values,
                        index,
                        columns,
                        aggfunc = np.mean,
                        fill_value = 0
                        )
    
    Parameters
    
    dataset - DataFrame, the data to calculate attribute aggregates with
    values - List of Strings, the attribute values to use for the aggregation
    index - List of Strings, the index of the initial pivot table
    columns - List of Strings, the columns of the initial pivot table
    aggfunc - Numpy.Module, the aggregation function to use, default is np.mean
    fill_value - Numeric, a default value to use for missing entries, default is 0
    
    Returns
    
    data_join
    
    Example
    
    gen_attr_agg_totals(dataset = base_agg_comp,
                        values = ['item_cnt_day'],
                        index = ['date_block_num'],
                        columns = ['shop_id'],
                        aggfunc = np.mean,
                        fill_value = 0
                        )
    
    """
    
    # extract the relevant input info
    feat_name = values[0]
    attr_agg = '_'.join(columns)
    join_cols = columns + index
    
    print('creating pivot table ...')
    
    # create pivot table for item sale by date block
    totals_table = pd.pivot_table(data = dataset, 
                                  values = values,
                                  index = index,
                                  columns = columns,
                                  aggfunc = aggfunc,
                                  fill_value = fill_value,
                                  dropna = True
                                  )
    shape = totals_table.shape
    print(shape)
    
    if index == ['date_block_num']:
        
        # overwite date block 34 with 0s
        totals_table.loc[34, :] = 0

    print('unstacking data ...')

    # unstack the attribute totals
    attr_name = '{}_total_{}'.format(attr_agg, feat_name)
    
    #if len(columns) == 1:
    unstack_data = totals_table.unstack()
    #elif len(columns) == 2:
    #    unstack_data = totals_table.unstack().unstack()
    attr_total_data = unstack_data.reset_index().drop(columns = ['level_0']).rename(columns = {0:attr_name})
    shape = attr_total_data.shape
    print(shape)
    
    print('joining back results ...')
     
    # store the total attribute
    data_join = dataset.merge(attr_total_data, on = join_cols, how = 'left')
    shape = data_join.shape
    print(shape)
    
    return data_join