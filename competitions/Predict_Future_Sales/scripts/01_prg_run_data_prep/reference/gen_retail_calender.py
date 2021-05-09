# -*- coding: utf-8 -*-
"""
Created on Sun May  9 15:52:48 2021

@author: oislen
"""

import pandas as pd
import clean_constants as clean_cons

def gen_retail_calender():
    
    """
    
    Generate Retail Calender Documentation
    
    Function Overview
    
    This functions generates a retail calender for all years between '2013-01-01 and '2015-11-30'
    
    Defaults
    
    gen_retail_calender()
    
    Parameters
    
    Returns
    
    agg_df - DataFrame, the retail calender
    
    Example
    
    gen_retail_calender()
    
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
