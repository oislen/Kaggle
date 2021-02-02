# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 20:33:00 2020

@author: oislen
"""

import pandas as pd
import reference.clean_constants as clean_cons
import reference.clean_utilities as utl
import pickle as pk

def agg_base_data(cons):

    """
    """
        
    print('Loading clean data ...')
    
    # load in the raw data
    item_categories, items, sales_train, sample_submission, shops, test = utl.load_files('clean', cons)
    
    print('aggregating base data ...')
    
    # want to aggregate to date_block_num, shop and product level
    sales_train = sales_train.sort_values(by = clean_cons.group_cols)
    agg_base = sales_train.groupby(clean_cons.group_cols, as_index = False).agg(clean_cons.agg_dict)
    
    # add ID column
    agg_base['ID'] = agg_base.index
    
    print('Create generalised test data ...')
    
    # create static columns
    test['year'] = 2015
    test['month'] = 11
    test['date_block_num'] = 34
    test['item_cnt_day'] = 0
    test['n_refund'] = 0
    test['n_sale'] = 0
    
    print('Pickling holdout shop_item_id combination...')
    
    test['shop_item_id'] = test['shop_id'].astype(str) + '_' + test['item_id'].astype(str)
    holdout_shop_item_id = test['shop_item_id'].unique()
    pk.dump(holdout_shop_item_id, open(cons.holdout_shop_item_id_comb, "wb"))
    
    print('Getting most recent sale price ...')

    # Generate most recent item price for test set 
    recent_price = utl.gen_most_recent_item_price(dataset = agg_base)
    join_cols = ['item_id']
    base_test_price = test.merge(recent_price, on = join_cols, how = 'left')
    
    # Fill in -999 default for missing prices 
    base_test_price['item_price'] = base_test_price['item_price'].fillna(-999)

    print('Concatenate Base and Test data ...')
    
    base_concat = pd.concat(objs = [agg_base, base_test_price], axis = 0, ignore_index = True)
    
    # note this impacts shop item id combination
    print('Removing duplicate shops ...')
    
    filt_shop_0 = base_concat['shop_id'] == 0
    filt_shop_1 = base_concat['shop_id'] == 1
    filt_shop_10 = base_concat['shop_id'] == 10
    
    base_concat.loc[filt_shop_0, 'shop_id'] = 57
    base_concat.loc[filt_shop_1, 'shop_id'] = 58
    base_concat.loc[filt_shop_10, 'shop_id'] = 11

    print('Calculate revenue ...')
    
    base_concat['revenue'] = base_concat['item_price'] * base_concat['item_cnt_day']
    
    print('Clip item count day totals to [0, 20] interval ...')
    
    base_concat['item_cnt_day'] = base_concat['item_cnt_day'].apply(lambda x: 0 if x < 0 else (20 if x >= 20 else x))
   
    # data shape
    shape = base_concat.shape
    
    print('Recast data ...')
    
    base_concat = utl.recast_df(dataset = base_concat)
    
    print('outputting aggregated base data {} ...'.format(shape))
    
    # output aggreated base data as feather file
    base_concat.to_feather(cons.base_agg_data_fpath)
    
    return
