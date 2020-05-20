# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 17:46:31 2020

@author: oislen
"""

import pandas as pd
import clean_constants as clean_cons
import numpy as np

def load_files(ver, cons):
    
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
                   lags = 12,
                   fill_value = 0
                   ):
    
    """
    
    gen_shift_attr(dataset, 
                   values = ['item_cnt_day'],
                   index = ['date_block_num'],
                   columns = ['shop_id', 'item_id']
                   )
    
    
    """
    
    # extract the relevant input info
    feat_name = values[0]
    join_cols = columns + index
    data_join = dataset[join_cols].copy(True)
    
    print('creating pivot table ...')
    
    # create pivot table for item sale by date block
    sales_table = pd.pivot_table(data = dataset, 
                                 values = values,
                                 index = index,
                                 columns = columns,
                                 fill_value = fill_value,
                                 aggfunc = np.sum,
                                 dropna = True
                                 )
    
    shape = sales_table.shape
    
    print(shape)
    
    # for each required shift
    for i in lags:
        
        print('Working on shift {} ...'.format(i))
        print('create shifts ...')
        
        # shift the data by i
        shift_data = sales_table.shift(i, fill_value = 0)
        shape = shift_data.shape     
        print(shape)
        
        print('unstacking data ...')
        
        # unstack the shifted attribute
        attr_name = '{}_shift_{}'.format(feat_name, i)
        unstack_data = shift_data.unstack()
        attr_shift_data = unstack_data.reset_index().drop(columns = ['level_0']).rename(columns = {0:attr_name})
        shape = attr_shift_data.shape
        print(shape)
        
        print('joining back results ...')
        
        # store the shifted attribute
        data_join = data_join.merge(attr_shift_data, on = join_cols, how = 'left')
        shape = data_join.shape
        print(shape)
        
    print('Consolidating all attributes ...')
    
    # set the join columns
    join_cols = columns + index
    
    # join shift attributes back to base data
    data_out = pd.merge(left = dataset,
                        right = data_join,
                        on = join_cols,
                        how = 'left'
                        )
    shape = data_out.shape
    print(shape)
    
    return data_out

def gen_most_recent_item_price(dataset):
    
    """
    
    Generate Most Recent Item Price
    
    """
    
    # Note: not all shops sell every item
    # Note: not all shops are the same price
    sort_cols = ['date_block_num', 'shop_id', 'item_id']
    group_cols = ['item_id']
    agg_dict = {'item_price':'last'}
    agg_base_sort = dataset.sort_values(by = sort_cols)
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
    
    # add in number of month days
    agg_df['days_of_month'] = agg_df['month'].map({1:31, 2:28, 3:31, 4:30, 5:31, 6:30, 7:31, 8:31, 9:30, 10:31, 11:30, 12:31})
    
    # insert date block num
    agg_df['date_block_num'] = agg_df.index
    
    # generate total holiday days
    agg_df['totalholidays'] = agg_df['n_weekenddays'] + agg_df['n_publicholidays']
    
    return agg_df






def backfill_attr(dataset, 
                  pivot_values, 
                  fillna = None,
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
                           aggfunc = np.sum,
                           fill_value = None,
                           dropna = True
                           )
    
    # if forward fill
    if ffill:
            
        # want to forward fill from given price to next price
        table = table.ffill(axis = 0)
        
    if fillna != None:
            
        # add in unknown default
        table = table.fillna(fillna)
        
    # double unstack to get data by year, month, item_id and shop_id
    unstack = table.stack().stack().reset_index()

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
    

def gen_attr_agg_totals(dataset, 
                        values, 
                        index, 
                        columns, 
                        fill_value = 0
                        ):
    
    """
    """
    
    # extract the relevant input info
    feat_name = values[0]
    attr_agg = columns[0]
    join_cols = columns + index
    
    print('creating pivot table ...')
    
    # create pivot table for item sale by date block
    totals_table = pd.pivot_table(data = dataset, 
                                  values = values,
                                  index = index,
                                  columns = columns,
                                  aggfunc = np.mean,
                                  fill_value = fill_value,
                                  dropna = True
                                  )
    shape = totals_table.shape
    print(shape)
    
    if index == ['date_block_num']:
        
        # overwite date block 34 with -999s
        totals_table.loc[34, :] = -999

    print('unstacking data ...')

    # unstack the attribute totals
    attr_name = '{}_total_{}'.format(attr_agg, feat_name)
    unstack_data = totals_table.unstack()
    attr_total_data = unstack_data.reset_index().drop(columns = ['level_0']).rename(columns = {0:attr_name})
    shape = attr_total_data.shape
    print(shape)
    
    print('joining back results ...')
     
    # store the total attribute
    data_join = dataset.merge(attr_total_data, on = join_cols, how = 'left')
    shape = data_join.shape
    print(shape)
    
    return data_join

def mean_encode(dataset, attr, tar):
        """
        """
        mu = dataset[tar].mean()
        cumsum = dataset.groupby(attr)[tar].cumsum() - dataset[tar]
        cumcnt = dataset.groupby(attr).cumcount()
        attr = cumsum/cumcnt
        attr = attr.fillna(mu)
        return attr