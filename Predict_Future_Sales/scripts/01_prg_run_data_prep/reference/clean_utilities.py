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
    attr_agg = '_'.join(columns)
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




def months_since_purchase(dataset, 
                          values, 
                          index, 
                          columns
                          ):
        
        data = dataset.copy(True)
        shape = data.shape
        print(shape)
        
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
        
        shape = out_data.shape
        print(shape)
        
        return out_data


def mean_encode(dataset, attr, tar):
        """
        """
        mu = dataset[tar].mean()
        cumsum = dataset.groupby(attr)[tar].cumsum() - dataset[tar]
        cumcnt = dataset.groupby(attr).cumcount()
        attr = cumsum/cumcnt
        attr = attr.fillna(mu)
        return attr


    
def extract_float_int_cols(data):
    
    data_cols = data.columns
    data_dtypes = data.dtypes
    
    print(data_dtypes.value_counts())
     
    # filter various int data types
    filt_int8 = data_dtypes ==  np.int8
    filt_int16 = data_dtypes ==  np.int16
    filt_int32 = data_dtypes ==  np.int32
    filt_int64 = data_dtypes ==  np.int64
    
    # filter various float data types
    filt_float16 = data_dtypes ==  np.float16
    filt_float32 = data_dtypes ==  np.float32
    filt_float64 = data_dtypes ==  np.float64
    
    # extract out the integer and float columns
    int_cols = data_cols[filt_int8 | filt_int16 | filt_int32 | filt_int64]
    float_cols = data_cols[filt_float16 | filt_float32 | filt_float64]
    
    return int_cols, float_cols
    
def recast_df(dataset):
    
    """
    """
    
    print('Copying data ...')
    
    data = dataset.copy(True)
    
    print('Converting non-null columns to int32s ...')
    
    # check missing values
    n_null = data.isnull().sum()
    nonull_cols = n_null[n_null == 0].index 
    
    # extract int and float columns
    int_cols, float_cols = extract_float_int_cols(data = data)
    
    # find non-null floats
    nonull_float_cols = [col for col in nonull_cols if col in float_cols]
    
    # check a random sample 1000 records for all '.0' decimals
    data_str = data[nonull_float_cols].sample(100000, random_state = 1234).astype(str)
    true_floats = data_str.apply(lambda x: x.str.contains('\.0').all(), axis = 0)
    
    # cast true floats to int
    cast_to_int_cols = true_floats[true_floats].index
    data[cast_to_int_cols] = data[cast_to_int_cols].astype(np.int32)
      
    # extract int and float columns
    int_cols, float_cols = extract_float_int_cols(data = data)
    
    print('Generating column min / max values ...')
    
    # get maximum values
    max_int = data[int_cols].max().rename('max').reset_index()
    min_int = data[int_cols].min().rename('min').reset_index()
    max_float = data[float_cols].max().rename('max').reset_index()
    min_float = data[float_cols].min().rename('min').reset_index()
    
    # join min max values
    join_int = min_int.merge(max_int, on = 'index', how = 'inner')
    join_float = min_float.merge(max_float, on = 'index', how = 'inner')
    
    # create dtype columns
    join_int['dtype'] = np.nan
    join_float['dtype'] = np.nan
    
    print('Defining dtype ranges for recasting ...')
    
    # get the data type limits
    lim_int8 = np.iinfo(np.int8)
    lim_int16 = np.iinfo(np.int16)
    lim_int32 = np.iinfo(np.int32)
    lim_int64 = np.iinfo(np.int64)
    #lim_float16 = np.finfo(np.float16)
    lim_float32 = np.finfo(np.float32)
    lim_float64 = np.finfo(np.float64)
    
    # get data type ranges
    range_int8 = (lim_int8.min, lim_int8.max) 
    range_int16 = (lim_int16.min, lim_int16.max) 
    range_int32 = (lim_int32.min, lim_int32.max) 
    range_int64 = (lim_int64.min, lim_int64.max) 
    #range_float16 = (lim_float16.min, lim_float16.max) 
    range_float32 = (lim_float32.min, lim_float32.max) 
    range_float64 = (lim_float64.min, lim_float64.max)  
    
    print('Finding optimal data type cast ...')
    
    # get apply min / max search
    join_int['dtype'] = join_int.apply(lambda x: 'int64' if x['min'] >= range_int64[0] and x['max'] <= range_int64[1] else x['dtype'], axis = 1)
    join_int['dtype'] = join_int.apply(lambda x: 'int32' if x['min'] >= range_int32[0] and x['max'] <= range_int32[1] else x['dtype'], axis = 1)
    join_int['dtype'] = join_int.apply(lambda x: 'int16' if x['min'] >= range_int16[0] and x['max'] <= range_int16[1] else x['dtype'], axis = 1)
    join_int['dtype'] = join_int.apply(lambda x: 'int8' if x['min'] >= range_int8[0] and x['max'] <= range_int8[1] else x['dtype'], axis = 1)
    
    # get apply min / max search
    join_float['dtype'] = join_float.apply(lambda x: 'float64' if x['min'] >= range_float64[0] and x['max'] <= range_float64[1] else x['dtype'], axis = 1)
    join_float['dtype'] = join_float.apply(lambda x: 'float32' if x['min'] >= range_float32[0] and x['max'] <= range_float32[1] else x['dtype'], axis = 1)
    #join_float['dtype'] = join_float.apply(lambda x: 'float16' if x['min'] >= range_float16[0] and x['max'] <= range_float16[1] else x['dtype'], axis = 1)
    
    print('Recasting data ...')
    
    # extract out the relevant data types
    filt_cast_float64 = join_float['dtype'] == 'float64'
    filt_cast_float32 = join_float['dtype'] == 'float32'
    cast_cols_float64 = join_float.loc[filt_cast_float64, 'index']
    cast_cols_float32 = join_float.loc[filt_cast_float32, 'index']
    
    # extract out the relevant data types
    filt_cast_int64 = join_int['dtype'] == 'int64'
    filt_cast_int32 = join_int['dtype'] == 'int32'
    filt_cast_int16 = join_int['dtype'] == 'int16'
    filt_cast_int8 = join_int['dtype'] == 'int8'
    cast_cols_int64 = join_int.loc[filt_cast_int64, 'index']
    cast_cols_int32 = join_int.loc[filt_cast_int32, 'index']
    cast_cols_int16 = join_int.loc[filt_cast_int16, 'index']
    cast_cols_int8 = join_int.loc[filt_cast_int8, 'index']
    
    # recast the data
    data[cast_cols_float64] = data[cast_cols_float64].astype(np.float64)
    data[cast_cols_float32] = data[cast_cols_float32].astype(np.float32)
    data[cast_cols_int64] = data[cast_cols_int64].astype(np.int64)
    data[cast_cols_int32] = data[cast_cols_int32].astype(np.int32)
    data[cast_cols_int16] = data[cast_cols_int16].astype(np.int16)
    data[cast_cols_int8] = data[cast_cols_int8].astype(np.int8)

    print(data.dtypes.value_counts())
        
    return data

def n_price_changes(dataset, 
                    values, 
                    index, 
                    columns
                    ):
    
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
    
