# -*- coding: utf-8 -*-
"""
Created on Sun May 10 19:15:38 2020

@author: oislen
"""

import pandas as pd
import reference.clean_utilities as utl

def prep_model_data(cons):
    
    """
    
    Prepare Model Data Documentation
    
    Function Overview
    
    """
    
    print('loading in base data ...')
    
    # load in the bases data file
    base_agg_comp = pd.read_feather(cons.base_agg_shft_fpath)
    
    # set function inputs for item count shift attributes
    index_shift = ['date_block_num']
    columns_shift = ['shop_id', 'item_id']
    lags = [1, 2, 3, 4, 6, 12]
    fill_na = 0
    
    ########################
    #-- Shift Attributes --#
    ########################

    print('Running shift attributes for item cnt ...')
    
    #-- Lag Item Cnt Shifts --#
    
    print('item_cnt_day')
    
    # create shift attributes
    base_agg_comp = utl.gen_shift_attr(dataset = base_agg_comp, 
                                       values = ['item_cnt_day'], 
                                       index = index_shift, 
                                       columns = columns_shift,
                                       lags = lags,
                                       fill_value = fill_na
                                       )
    
    #-- Lag Shop Total Shifts --#
    
    print('shop_id_total_item_cnt_day')
    
    # create shift attributes
    base_agg_comp = utl.gen_shift_attr(dataset = base_agg_comp, 
                                       values = ['shop_id_total_item_cnt_day'], 
                                       index = index_shift, 
                                       columns = columns_shift,
                                       lags = lags,
                                       fill_value = fill_na
                                       )
    
    #--Lag Item Total Shifts --#
    
    print('item_id_total_item_cnt_day')
    
    # create shift attributes
    base_agg_comp = utl.gen_shift_attr(dataset = base_agg_comp, 
                                       values = ['item_id_total_item_cnt_day'], 
                                       index = index_shift, 
                                       columns = columns_shift,
                                       lags = lags,
                                       fill_value = fill_na
                                       )
     
    #-- Lag Item Price --#
    
    print('item_price')
    
    # create shift attributes
    base_agg_comp = utl.gen_shift_attr(dataset = base_agg_comp, 
                                       values = ['item_price'], 
                                       index = index_shift, 
                                       columns = columns_shift,
                                       lags = [1],
                                       fill_value = fill_na
                                       )
    
    
    #-- Lag Revenue --#
    
    print('revenue')
    
    # create shift attributes
    base_agg_comp = utl.gen_shift_attr(dataset = base_agg_comp, 
                                       values = ['revenue'], 
                                       index = index_shift, 
                                       columns = columns_shift,
                                       lags = [1],
                                       fill_value = fill_na
                                       )
    
    #-- Lag Shop id Cat id Total --#
    
    print('item_category_id_total_item_cnt_day')
    
    # create shift attributes
    
    base_agg_comp = utl.gen_shift_attr(dataset = base_agg_comp, 
                                       values = ['item_category_id_total_item_cnt_day'], 
                                       index = index_shift, 
                                       columns = columns_shift,
                                       lags = [1],
                                       fill_value = fill_na
                                       )
    
    print('shop_id_item_category_id_total_item_cnt_day')
    
    # create shift attributes
    base_agg_comp = utl.gen_shift_attr(dataset = base_agg_comp, 
                                       values = ['shop_id_item_category_id_total_item_cnt_day'], 
                                       index = index_shift, 
                                       columns = columns_shift,
                                       lags = [1],
                                       fill_value = fill_na
                                       )
        
    print('city_enc_total_item_cnt_day')
    
    # create shift attributes
    base_agg_comp = utl.gen_shift_attr(dataset = base_agg_comp, 
                                       values = ['city_enc_total_item_cnt_day'], 
                                       index = index_shift, 
                                       columns = columns_shift,
                                       lags = [1],
                                       fill_value = fill_na
                                       )
            
    print('item_id_city_enc_total_item_cnt_day')
    
    # create shift attributes
    base_agg_comp = utl.gen_shift_attr(dataset = base_agg_comp, 
                                       values = ['item_id_city_enc_total_item_cnt_day'], 
                                       index = index_shift, 
                                       columns = columns_shift,
                                       lags = [1],
                                       fill_value = fill_na
                                       )
    
    #print('Replace -999s with missing values ...')
    
    #base_agg_comp = base_agg_comp.replace(-999, np.nan)
    
    print('Removing 1st year of data due to lagged attributes ...')
    
    filt_1st_year = base_agg_comp['date_block_num'] >= 12
    base_agg_comp = base_agg_comp[filt_1st_year]
    shape = base_agg_comp.shape
    
    
    print('Create delta attributes ...')
    
    # TODO: add delta revenue
    base_agg_comp['delta_item_price'] = base_agg_comp['item_price'] - base_agg_comp['item_price_shift_1']
    base_agg_comp['delta_item_cnt_day_1_2'] = base_agg_comp['item_cnt_day_shift_1'] - base_agg_comp['item_cnt_day_shift_2']
    base_agg_comp['delta_item_cnt_day_1_3'] = base_agg_comp['item_cnt_day_shift_1'] - base_agg_comp['item_cnt_day_shift_3']
    
    print('Create interaction attributes ...')
    
    #base_agg_comp['shop_id_total_item_cnt_day_shift_1_x_item_id_total_item_cnt_day_shift_1'] = base_agg_comp['shop_id_total_item_cnt_day_shift_1'] - base_agg_comp['item_id_total_item_cnt_day_shift_1']
    #base_agg_comp['shop_id_total_item_cnt_day_shift_2_x_item_id_total_item_cnt_day_shift_2'] = base_agg_comp['shop_id_total_item_cnt_day_shift_2'] - base_agg_comp['item_id_total_item_cnt_day_shift_2']
    #base_agg_comp['shop_id_total_item_cnt_day_shift_3_x_item_id_total_item_cnt_day_shift_3'] = base_agg_comp['shop_id_total_item_cnt_day_shift_3'] - base_agg_comp['item_id_total_item_cnt_day_shift_3']
    
    print('Create proportion attributes ...')
    
    base_agg_comp['item_cnt_day_shift_1_div_shop_id_total_item_cnt_day_shift_1'] = (base_agg_comp['item_cnt_day_shift_1'] / base_agg_comp['shop_id_total_item_cnt_day_shift_1']).fillna(0)
    base_agg_comp['item_cnt_day_shift_2_div_shop_id_total_item_cnt_day_shift_2'] = (base_agg_comp['item_cnt_day_shift_2'] / base_agg_comp['shop_id_total_item_cnt_day_shift_2']).fillna(0)
    base_agg_comp['item_cnt_day_shift_3_div_shop_id_total_item_cnt_day_shift_3'] = (base_agg_comp['item_cnt_day_shift_3'] / base_agg_comp['shop_id_total_item_cnt_day_shift_3']).fillna(0)
    base_agg_comp['item_cnt_day_shift_4_div_shop_id_total_item_cnt_day_shift_4'] = (base_agg_comp['item_cnt_day_shift_4'] / base_agg_comp['shop_id_total_item_cnt_day_shift_4']).fillna(0)
    base_agg_comp['item_cnt_day_shift_6_div_shop_id_total_item_cnt_day_shift_6'] = (base_agg_comp['item_cnt_day_shift_6'] / base_agg_comp['shop_id_total_item_cnt_day_shift_6']).fillna(0)
    base_agg_comp['item_cnt_day_shift_12_div_shop_id_total_item_cnt_day_shift_12'] = (base_agg_comp['item_cnt_day_shift_12'] / base_agg_comp['shop_id_total_item_cnt_day_shift_12']).fillna(0)
    base_agg_comp['item_cnt_day_shift_1_div_item_id_total_item_cnt_day_shift_1'] = (base_agg_comp['item_cnt_day_shift_1'] / base_agg_comp['item_id_total_item_cnt_day_shift_1']).fillna(0)
    base_agg_comp['item_cnt_day_shift_2_div_item_id_total_item_cnt_day_shift_2'] = (base_agg_comp['item_cnt_day_shift_2'] / base_agg_comp['item_id_total_item_cnt_day_shift_2']).fillna(0)
    base_agg_comp['item_cnt_day_shift_3_div_item_id_total_item_cnt_day_shift_3'] = (base_agg_comp['item_cnt_day_shift_3'] / base_agg_comp['item_id_total_item_cnt_day_shift_3']).fillna(0)
    base_agg_comp['item_cnt_day_shift_4_div_item_id_total_item_cnt_day_shift_4'] = (base_agg_comp['item_cnt_day_shift_4'] / base_agg_comp['item_id_total_item_cnt_day_shift_4']).fillna(0)
    base_agg_comp['item_cnt_day_shift_6_div_item_id_total_item_cnt_day_shift_6'] = (base_agg_comp['item_cnt_day_shift_6'] / base_agg_comp['item_id_total_item_cnt_day_shift_6']).fillna(0)
    base_agg_comp['item_cnt_day_shift_12_div_item_id_total_item_cnt_day_shift_12'] = (base_agg_comp['item_cnt_day_shift_12'] / base_agg_comp['item_id_total_item_cnt_day_shift_12']).fillna(0)

    print('Subsetting required columns ...')
    
    # set columns to drop
    data_cols = base_agg_comp.columns
    drop_cols = ['shop_item_id', 'item_cat', 'item_cat_sub', 'city', 'revenue', ]
    sub_cols = data_cols.drop(drop_cols)
    model_data = base_agg_comp[sub_cols]
    model_data = model_data.reset_index(drop = True)
    
    print('Recasting data ...')
    
    model_data = utl.recast_df(dataset = model_data)
    
    shape = model_data.shape
    
    print('outputting model data {} ....'.format(shape))
    
    # output the model data
    model_data.to_feather(cons.model_data_fpath)

    return
