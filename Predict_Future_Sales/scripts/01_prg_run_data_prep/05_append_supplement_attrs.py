# -*- coding: utf-8 -*-
"""
Created on Sun May 10 18:06:01 2020

@author: oislen
"""

import pandas as pd
import reference.clean_utilities as utl

def append_supplement_attrs(cons):
    
    
    """
    
    Append Supplementary Attributes Documenation
    
    Function Overview
    
    This function appends the
    
    """

    # load in the raw data
    item_categories, items, sales_train, sample_submission, shops, test = utl.load_files('clean', cons)
    
    # output aggreated base data as feather file
    base_agg_comp = pd.read_feather(cons.base_agg_shft_fpath)
    shape = base_agg_comp.shape
    print(shape)
    
    print('Extract the price info ...')
    
    # extract price information
    base_agg_comp['price_decimal'] = base_agg_comp['item_price'].astype(str).str.extract('\d+\.(\d*)')[0].astype(float)
    base_agg_comp['price_decimal_len'] = base_agg_comp['item_price'].astype(str).str.extract('\d+\.(\d*)')[0].str.len()
    
    print('Joining clean data ...')
    
    # join all data sets together
    base_agg_comp = base_agg_comp.merge(items, on = 'item_id', how = 'left')
    base_agg_comp = base_agg_comp.merge(item_categories, on = 'item_category_id', how = 'left')
    base_agg_comp = base_agg_comp.merge(shops, on = 'shop_id', how = 'left')
    
    print('Subset required columns ...')
    
    base_cols = base_agg_comp.columns
    drop_cols = ['item_name', 'item_category_name', 'shop_name', 'shop_quotes', 'shop_brackets', 'shop_smooth']
    sub_cols = base_cols[~base_cols.isin(drop_cols)]
    base_agg_comp = base_agg_comp[sub_cols]
    
    print('Generate calendar days ...')
    
    retail_calander = utl.gen_retail_calender()
    join_cols = ['date_block_num']
    base_agg_comp = base_agg_comp.merge(retail_calander, on = join_cols, how = 'left')
    
    shape = base_agg_comp.shape
    
    print('Create delta attributes ...')
    
    base_agg_comp['delta_item_price'] = base_agg_comp['item_price'] - base_agg_comp['item_price_shift_1']
    base_agg_comp['delta_item_cnt_day_1_2'] = base_agg_comp['item_cnt_day_shift_1'] - base_agg_comp['item_cnt_day_shift_2']
    base_agg_comp['delta_item_cnt_day_1_3'] = base_agg_comp['item_cnt_day_shift_1'] - base_agg_comp['item_cnt_day_shift_3']
    base_agg_comp['delta_item_cnt_day_1_4'] = base_agg_comp['item_cnt_day_shift_1'] - base_agg_comp['item_cnt_day_shift_4']
    base_agg_comp['delta_item_cnt_day_1_6'] = base_agg_comp['item_cnt_day_shift_1'] - base_agg_comp['item_cnt_day_shift_6']
    base_agg_comp['delta_item_cnt_day_1_12'] = base_agg_comp['item_cnt_day_shift_1'] - base_agg_comp['item_cnt_day_shift_12']
    
    print('Create interaction attributes ...')
    
    base_agg_comp['shop_id_total_item_cnt_day_shift_1_x_item_id_total_item_cnt_day_shift_1'] = base_agg_comp['shop_id_total_item_cnt_day_shift_1'] - base_agg_comp['item_id_total_item_cnt_day_shift_1']
    base_agg_comp['shop_id_total_item_cnt_day_shift_2_x_item_id_total_item_cnt_day_shift_2'] = base_agg_comp['shop_id_total_item_cnt_day_shift_2'] - base_agg_comp['item_id_total_item_cnt_day_shift_2']
    base_agg_comp['shop_id_total_item_cnt_day_shift_3_x_item_id_total_item_cnt_day_shift_3'] = base_agg_comp['shop_id_total_item_cnt_day_shift_3'] - base_agg_comp['item_id_total_item_cnt_day_shift_3']
    base_agg_comp['shop_id_total_item_cnt_day_shift_4_x_item_id_total_item_cnt_day_shift_4'] = base_agg_comp['shop_id_total_item_cnt_day_shift_4'] - base_agg_comp['item_id_total_item_cnt_day_shift_4']
    base_agg_comp['shop_id_total_item_cnt_day_shift_6_x_item_id_total_item_cnt_day_shift_6'] = base_agg_comp['shop_id_total_item_cnt_day_shift_6'] - base_agg_comp['item_id_total_item_cnt_day_shift_6']
    base_agg_comp['shop_id_total_item_cnt_day_shift_12_x_item_id_total_item_cnt_day_shift_12'] = base_agg_comp['shop_id_total_item_cnt_day_shift_12'] - base_agg_comp['item_id_total_item_cnt_day_shift_12']
    
    print('Create proportion attributes ...')
    
    base_agg_comp['item_cnt_day_shift_1_div_shop_id_total_item_cnt_day_shift_1'] = base_agg_comp['item_cnt_day_shift_1'] - base_agg_comp['shop_id_total_item_cnt_day_shift_1']
    base_agg_comp['item_cnt_day_shift_1_div_item_id_total_item_cnt_day_shift_1'] = base_agg_comp['item_cnt_day_shift_1'] - base_agg_comp['item_id_total_item_cnt_day_shift_1']
    
    print('Outputting supplementary data {} ...'.format(shape))
    
    # output file as a feather file
    base_agg_comp.to_feather(cons.base_agg_supp_fpath)
    
    return
