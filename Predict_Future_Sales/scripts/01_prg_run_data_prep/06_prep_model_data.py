# -*- coding: utf-8 -*-
"""
Created on Sun May 10 19:15:38 2020

@author: oislen
"""

import pandas as pd
import numpy as np
import reference.clean_utilities as utl
from sklearn import preprocessing

def prep_model_data(cons):
    
    """
    
    Prepare Model Data Documentation
    
    Function Overview
    
    """
    
    print('loading in base data ...')
    
    # load in the bases data file
    base = pd.read_feather(cons.base_agg_shft_fpath)
    
    dataset = base
    values = ['item_cnt_day']
    index = ['date_block_num']
    columns = ['shop_id', 'item_id']
    
    def months_since_purchase(dataset, values, index, columns):
        
        data = dataset.copy(True)
        
        attr = '_'.join(columns)
        join_cols = index + columns
        
        print('Creating pivot table ...')
        
        # create pivot table for item sale by date block
        sales_table = pd.pivot_table(data = data, 
                                     values = values,
                                     index = index,
                                     columns = columns,
                                     fill_value = 0,
                                     aggfunc = np.sum,
                                     dropna = True
                                     )
        
        print('Formatting table ...')
        
        # convert all postive cases to one and all zeros to missing
        filt_sales = sales_table >= 1
        sales_table[filt_sales] = 1
        tab = sales_table[filt_sales]
        
        print('Calculating months since first and last purchase ...')
        
        # calculate months since first purchase
        msfp = tab.ffill().notnull().cumsum()
        mslp = tab.bfill().isnull().cumsum()
        
        msfp_unstack = msfp.unstack().reset_index().drop(columns = ['level_0']).rename(columns = {0:'{}_months_first_rec'.format(attr)})
        mslp_unstack = mslp.unstack().reset_index().drop(columns = ['level_0']).rename(columns = {0:'{}_months_last_rec'.format(attr)})
        
        print('Joining back results ...')
        
        # join together the months since purchase attributes 
        msp_unstack = pd.merge(left = msfp_unstack, right = mslp_unstack, how = 'inner', on = join_cols)
        
        # join back to data
        out_data = data.merge(msp_unstack, how = 'left', on = join_cols)
        
        return out_data
        
    msp = months_since_purchase(dataset, values, index, columns)
    
    sales_tab = sales_table[sales_table[filt_sales] == 1]
    sales_tab.apply(lambda x: x.idxmax())
    
    # select first puchase month
    sales_table.apply(lambda x: x.idxmax())
    sales_table.apply(lambda x: x 
    

    print('Subsetting required columns ...')
    
    # set columns to drop
    data_cols = base.columns
    drop_cols = ['shop_item_id', 'item_cat', 'item_cat_sub', 'city', 'revenue', ]
    sub_cols = data_cols.drop(drop_cols)
    model_data = base[sub_cols]
    
    # TODO: add down casting here to float16 and int8
    
    """
    data_cols = model_data.columns
    data_dtypes = model_data.dtypes
    data_dtypes.value_counts()
    
    int32_cols = data_cols[data_dtypes == np.int32]
    int64_cols = data_cols[data_dtypes == np.int64]
    float32_cols = data_cols[data_dtypes == np.float64]
    
    model_data[int32_cols] = model_data[int32_cols].astype(np.int8)
    model_data[int64_cols] = model_data[int64_cols].astype(np.int8)
    model_data[float32_cols] = model_data[float32_cols].astype(np.float32)
    
    model_data.dtypes.value_counts()
    """

    shape = model_data.shape
    
    print('outputting model data {} ....'.format(shape))
    
    # output the model data
    model_data.to_feather(cons.model_data_fpath)

    return
