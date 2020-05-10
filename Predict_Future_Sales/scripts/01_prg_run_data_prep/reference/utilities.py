# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 17:46:31 2020

@author: oislen
"""

import pandas as pd
import file_constants as cons
import clean_constants as clean_cons
import numpy as np
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

def gen_most_recent_item_price(dataset):
    
    """
    
    Generate Most Recent Item Price
    
    """
    
    sort_cols = ['year', 'month', 'shop_id', 'item_id']
    group_cols = ['item_id']
    agg_base_sort = dataset.sort_values(by = sort_cols)
    agg_dict = {'item_price':'last'}
    recent_price = agg_base_sort.groupby(group_cols, as_index = False).agg(agg_dict)
    
    return recent_price
    

def gen_retail_calender():
    
    """
    
    Generate Retail Calender
    
    """
    
    calender = pd.date_range(start = '2013-01-01', end = '2015-11-30')
    calender_df = pd.DataFrame(calender, columns = ['date'])
    
    # extract year and month
    calender_df['year'] = calender_df['date'].dt.year
    calender_df['month'] = calender_df['date'].dt.month

    # add weekdays and weekends 
    calender_df['dayofweek'] = calender_df['date'].apply(lambda x: x.dayofweek)
    calender_df['n_weekenddays'] = calender_df['dayofweek'].isin([5, 6]).astype(int)
    
    # add public holidays
    public_holidays_list = [hol for key, val in clean_cons.russian_holidays.items() for day, hol in val.items()]
    public_holidays_ranges = [pd.Series(pd.date_range(hol[0], hol[1])) for hol in public_holidays_list]
    public_holidays_series = pd.concat(objs = public_holidays_ranges, ignore_index = True)
    calender_df['n_publicholidays'] = calender_df['date'].isin(public_holidays_series).astype(int)

    # aggregate up to year month level
    group_cols = ['year', 'month']
    agg_dict = {'n_weekenddays':'sum', 'n_publicholidays':'sum'}
    agg_df = calender_df.groupby(group_cols, as_index = False).agg(agg_dict)
    
    # generate total holiday days
    agg_df['totalholidays'] = agg_df['n_weekenddays'] + agg_df['n_publicholidays']
    
    return agg_df






def backfill_attr(dataset, 
                  pivot_values, 
                  fillna,
                  pivot_index = ['year', 'month'], 
                  pivot_columns = ['shop_id', 'item_id'], 
                  ffill = False
                  ):
    
    """
    
    utl.backfill_attr(dataset = agg_base, 
                      pivot_values = ['item_price'], 
                      fillna = -999,
                      pivot_index = ['year', 'month'], 
                      pivot_columns = ['shop_id', 'item_id'], 
                      ffill = True
                      )
    
    """
        
    print('Back filling {} ...'.format(pivot_values[0]))
    
    # set up for finding price of an item
    table = pd.pivot_table(data = dataset, 
                           values = pivot_values,
                           index = pivot_index,
                           columns = pivot_columns,
                           dropna = True
                           )
    
    # if forward fill
    if ffill:
            
        # want to forward fill from given price to next price
        table = table.ffill(axis = 0)
        
    # add in unknown default
    table = table.fillna(fillna)
    
    # double unstack to get data by year, month, item_id and shop_id
    unstack = table.stack().stack().reset_index()
    
    # filter out future dates
    date_filt = (unstack['year'] == 2015) & (unstack['month'] == 12)
    unstack = unstack.loc[~date_filt, :]
    
    return unstack

def fill_id(dataset, fill_type, split, fillna = -999):
        """
        """
        data = dataset.copy(True)
        filt_split = data['data_split'] == split
        nrows = filt_split.sum()
        if fill_type == 'range':
            data.loc[filt_split, 'ID'] = np.arange(nrows)
        elif fill_type == 'value':
            data.loc[filt_split, 'ID'] = data.loc[filt_split, 'ID'].fillna(fillna)
        return data
    