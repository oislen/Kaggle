# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 17:46:31 2020

@author: oislen
"""

import pandas as pd
import file_constants as cons

def load_files(ver):
    
    """
    Loads in all files
    """
    
    if ver == 'raw':
        encoding = 'latin1'
        item_categories = pd.read_csv(cons.item_categories_fpath, encoding = encoding)
        items = pd.read_csv(cons.items_fpath)
        sales_train = pd.read_csv(cons.sales_train_fpath)
        sample_submission = pd.read_csv(cons.sample_submission_fpath) 
        shops = pd.read_csv(cons.shops_fpath, encoding = encoding)
        test = pd.read_csv(cons.test_fpath)
        
    elif ver == 'clean':
        
        items = pd.read_feather(cons.items_clean_fpath)
        sample_submission = pd.read_feather(cons.sample_submission_clean_fpath)
        sales_train = pd.read_feather(cons.sales_train_clean_fpath)
        test = pd.read_feather(cons.test_clean_fpath)
        item_categories = pd.read_feather(cons.item_categories_clean_fpath)
        shops = pd.read_feather(cons.shops_clean_fpath)
        
    return item_categories, items, sales_train, sample_submission, shops, test
    

def gen_shift_attr(dataset, 
                   values, 
                   index, 
                   columns, 
                   fill_value = 0
                   ):
    
    """
    
    gen_shift_attr(dataset, 
                   values = ['item_cnt_day'],
                   index = ['date_block_num'],
                   columns = ['shop_id', 'item_id']
                   )
    
    
    """
    
    feat_name = values[0]
    join_cols = columns + index
    data_join = dataset[join_cols].copy(True)
    
    print('creating pivot table ...')
    # create pivot table for item sale by date block
    sales_table = pd.pivot_table(data = dataset, 
                                 values = values,
                                 index = index,
                                 columns = columns,
                                 fill_value = 0,
                                 dropna = False
                                 )
    
    print('Removing all zero columns ...')
    # remove the columns all mising 
    cols = sales_table.columns
    non_zero_cols = ~(sales_table == 0).all()
    sales_table_filt = sales_table[cols[non_zero_cols]]
    nrow = sales_table.shape[0]
    
    for i in range(1, nrow + 1):
        print('~~~~~ working on {} ...'.format(i))
        print('create shifts ...')
        shift_data = sales_table_filt.shift(i)
        print('unstacking data ...')
        attr_name = '{}_shift_{}'.format(feat_name, i)
        unstack_data = shift_data.unstack()
        attr_shift_data = unstack_data.reset_index().drop(columns = ['level_0']).rename(columns = {0:attr_name})
        print('joining back results ...')
        data_join = data_join.merge(attr_shift_data, on = join_cols, how = 'left')
    
    # incorportate write here
    
    return data_join