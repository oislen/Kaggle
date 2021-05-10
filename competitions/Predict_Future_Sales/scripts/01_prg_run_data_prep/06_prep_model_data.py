# -*- coding: utf-8 -*-
"""
Created on Sun May 10 19:15:38 2020

@author: oislen
"""

import pandas as pd
import reference.clean_constants as clean_cons
from sklearn.preprocessing import StandardScaler
from reference.gen_shift_attr import gen_shift_attr
from reference.mean_encode import mean_encode
from reference.recast_df import recast_df
import pickle as pk

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
    lags = [1, 2, 3, 4]
    fill_na = 0
    
    ########################
    #-- Shift Attributes --#
    ########################

    print('Running shift attributes for item cnt ...')
    
    #-- Lag Item Cnt Shifts --#
    
    print('item_cnt_day')
    
    # create shift attributes
    base_agg_comp = gen_shift_attr(dataset = base_agg_comp, 
                                   values = ['item_cnt_day'], 
                                   index = index_shift, 
                                   columns = columns_shift,
                                   lags = lags,
                                   fill_value = fill_na
                                   )
    
    #-- Lag Shop Total Shifts --#
    
    print('shop_id_total_item_cnt_day')
    
    # create shift attributes
    base_agg_comp = gen_shift_attr(dataset = base_agg_comp, 
                                   values = ['shop_id_total_item_cnt_day'], 
                                   index = index_shift, 
                                   columns = columns_shift,
                                   lags = lags,
                                   fill_value = fill_na
                                   )
    
    #--Lag Item Total Shifts --#
    
    print('item_id_total_item_cnt_day')
    
    # create shift attributes
    base_agg_comp = gen_shift_attr(dataset = base_agg_comp, 
                                   values = ['item_id_total_item_cnt_day'], 
                                   index = index_shift, 
                                   columns = columns_shift,
                                   lags = lags,
                                   fill_value = fill_na
                                   )
     
    #-- Lag Item Price --#
    
    print('item_price')
    
    # create shift attributes
    base_agg_comp = gen_shift_attr(dataset = base_agg_comp, 
                                   values = ['item_price'], 
                                   index = index_shift, 
                                   columns = columns_shift,
                                   lags = [1],
                                   fill_value = fill_na
                                   )
    
    
    #-- Lag Revenue --#
    
    print('revenue')
    
    # create shift attributes
    base_agg_comp = gen_shift_attr(dataset = base_agg_comp, 
                                   values = ['revenue'], 
                                   index = index_shift, 
                                   columns = columns_shift,
                                   lags = [1],
                                   fill_value = fill_na
                                   )
    
    #-- Lag Shop id Cat id Total --#
    
    print('item_category_id_total_item_cnt_day')
    
    # create shift attributes
    
    base_agg_comp = gen_shift_attr(dataset = base_agg_comp, 
                                   values = ['item_category_id_total_item_cnt_day'], 
                                   index = index_shift, 
                                   columns = columns_shift,
                                   lags = [1],
                                   fill_value = fill_na
                                   )
    
    print('shop_id_item_category_id_total_item_cnt_day')
    
    # create shift attributes
    base_agg_comp = gen_shift_attr(dataset = base_agg_comp, 
                                   values = ['shop_id_item_category_id_total_item_cnt_day'], 
                                   index = index_shift, 
                                   columns = columns_shift,
                                   lags = [1],
                                   fill_value = fill_na
                                   )
        
    print('city_enc_total_item_cnt_day')
    
    # create shift attributes
    base_agg_comp = gen_shift_attr(dataset = base_agg_comp, 
                                   values = ['city_enc_total_item_cnt_day'], 
                                   index = index_shift, 
                                   columns = columns_shift,
                                   lags = [1],
                                   fill_value = fill_na
                                   )
            
    print('item_id_city_enc_total_item_cnt_day')
    
    # create shift attributes
    base_agg_comp = gen_shift_attr(dataset = base_agg_comp, 
                                   values = ['item_id_city_enc_total_item_cnt_day'], 
                                   index = index_shift, 
                                   columns = columns_shift,
                                   lags = [1],
                                   fill_value = fill_na
                                   )
    

    #table.apply(lambda x: x.rank(axis = 0, method = 'first'), axis = 1)
    
    print('Removing 1st year of data due to lagged attributes ...')
    
    filt_1st_year = base_agg_comp['date_block_num'] >= 4
    base_agg_comp = base_agg_comp[filt_1st_year]
    shape = base_agg_comp.shape
    
    
    print('Create delta attributes ...')
    
    # TODO: add delta revenue
    base_agg_comp['delta_item_price'] = base_agg_comp['item_price'] - base_agg_comp['item_price_shift_1']
    base_agg_comp['delta_item_cnt_day_1_2'] = base_agg_comp['item_cnt_day_shift_1'] - base_agg_comp['item_cnt_day_shift_2']
    base_agg_comp['delta_item_cnt_day_2_3'] = base_agg_comp['item_cnt_day_shift_2'] - base_agg_comp['item_cnt_day_shift_3']
    base_agg_comp['delta_item_cnt_day_3_4'] = base_agg_comp['item_cnt_day_shift_3'] - base_agg_comp['item_cnt_day_shift_4']
    
    print('Create proportion attributes ...')
    
    base_agg_comp['item_cnt_day_shift_1_div_shop_id_total_item_cnt_day_shift_1'] = (base_agg_comp['item_cnt_day_shift_1'] / base_agg_comp['shop_id_total_item_cnt_day_shift_1']).fillna(0)
    base_agg_comp['item_cnt_day_shift_2_div_shop_id_total_item_cnt_day_shift_2'] = (base_agg_comp['item_cnt_day_shift_2'] / base_agg_comp['shop_id_total_item_cnt_day_shift_2']).fillna(0)
    base_agg_comp['item_cnt_day_shift_3_div_shop_id_total_item_cnt_day_shift_3'] = (base_agg_comp['item_cnt_day_shift_3'] / base_agg_comp['shop_id_total_item_cnt_day_shift_3']).fillna(0)
    base_agg_comp['item_cnt_day_shift_4_div_shop_id_total_item_cnt_day_shift_4'] = (base_agg_comp['item_cnt_day_shift_4'] / base_agg_comp['shop_id_total_item_cnt_day_shift_4']).fillna(0)
    base_agg_comp['item_cnt_day_shift_1_div_item_id_total_item_cnt_day_shift_1'] = (base_agg_comp['item_cnt_day_shift_1'] / base_agg_comp['item_id_total_item_cnt_day_shift_1']).fillna(0)
    base_agg_comp['item_cnt_day_shift_2_div_item_id_total_item_cnt_day_shift_2'] = (base_agg_comp['item_cnt_day_shift_2'] / base_agg_comp['item_id_total_item_cnt_day_shift_2']).fillna(0)
    base_agg_comp['item_cnt_day_shift_3_div_item_id_total_item_cnt_day_shift_3'] = (base_agg_comp['item_cnt_day_shift_3'] / base_agg_comp['item_id_total_item_cnt_day_shift_3']).fillna(0)
    base_agg_comp['item_cnt_day_shift_4_div_item_id_total_item_cnt_day_shift_4'] = (base_agg_comp['item_cnt_day_shift_4'] / base_agg_comp['item_id_total_item_cnt_day_shift_4']).fillna(0)

    print('Mean encoding data ...')
    
    base_agg_comp['date_block_num_mean_enc'] = mean_encode(dataset = base_agg_comp, attr = ['date_block_num'], tar = 'item_cnt_day')
    base_agg_comp['shop_id_mean_enc'] = mean_encode(dataset = base_agg_comp, attr = ['shop_id'], tar = 'item_cnt_day')
    base_agg_comp['item_id_mean_enc'] = mean_encode(dataset = base_agg_comp, attr = ['item_id'], tar = 'item_cnt_day')
    base_agg_comp['shop_item_id_mean_enc'] = mean_encode(dataset = base_agg_comp, attr = ['shop_item_id'], tar = 'item_cnt_day')
    base_agg_comp['item_category_id_mean_enc'] = mean_encode(dataset = base_agg_comp, attr = ['item_category_id'], tar = 'item_cnt_day')
    base_agg_comp['item_cat_mean_enc'] = mean_encode(dataset = base_agg_comp, attr = ['item_cat'], tar = 'item_cnt_day')
    base_agg_comp['item_cat_sub_mean_enc'] = mean_encode(dataset = base_agg_comp, attr = ['item_cat_sub'], tar = 'item_cnt_day')
    base_agg_comp['city_mean_enc'] = mean_encode(dataset = base_agg_comp, attr = ['city'], tar = 'item_cnt_day')
    base_agg_comp['year_mean_enc'] = mean_encode(dataset = base_agg_comp, attr = ['year'], tar = 'item_cnt_day')
    base_agg_comp['month_mean_enc'] = mean_encode(dataset = base_agg_comp, attr = ['month'], tar = 'item_cnt_day')
    
    """
    print('Remove all items with no historic sell price from training set ...')
    
    # load in pickled holdout item shop id combination
    #holdout_shop_item_id_comb = pk.load(open(cons.holdout_shop_item_id_comb, 'rb'))
    base_agg_comp['shop_item_id'].nunique()
    
    # create filters for item price and holdout shop item combination
    price_tab = pd.pivot_table(data = base_agg_comp,
                               index = 'date_block_num',
                               columns = ['shop_id', 'item_id'],
                               values = 'item_price'
                               )
    all_zero_price = (price_tab == 0).all()
    keep_shop_item_comb = all_zero_price[all_zero_price].reset_index().drop(columns = 0)
    keep_shop_item_comb_series = keep_shop_item_comb['shop_id'].astype(str) + '_' + keep_shop_item_comb['item_id'].astype(str)
    
    filt_shop_item_id = base_agg_comp['shop_item_id'].isin(keep_shop_item_comb_series)
    filt_default_price = base_agg_comp['item_price'] == 0
    
    base_agg_comp = base_agg_comp[~filt_default_price | filt_shop_item_id]
    """
    
    print('Subsetting required columns ...')
    
    # set columns to drop
    data_cols = base_agg_comp.columns
    drop_cols = ['shop_item_id', 
                 'item_cat', 
                 'item_cat_sub', 
                 'city',
                 'revenue',
                 # dodgey attributes:
                 'delta_item_price', 
                 'shop_id_item_id_months_last_rec', 
                 'item_price', 
                 'n_price_changes', 
                 'shop_id_item_id_months_first_rec'
                 ]
    sub_cols = data_cols.drop(drop_cols)
    model_data = base_agg_comp[sub_cols]
    model_data = model_data.reset_index(drop = True)
    
    """
    print('Normalise data ...')
    
    ignore_cols = clean_cons.norm_ign_cols
    norm_cols = [col for col in model_data.columns if col not in ignore_cols]
    scaler = StandardScaler()
    scaler.fit(model_data[norm_cols])
    model_data[norm_cols] = scaler.transform(X = model_data[norm_cols])
    """
    
    print('Recasting data ...')
    
    model_data = recast_df(dataset = model_data)
    
    shape = model_data.shape
    
    print('outputting model data {} ....'.format(shape))
    
    # output the model data
    model_data.to_feather(cons.model_data_fpath)

    return
