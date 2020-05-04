# -*- coding: utf-8 -*-
"""
Created on Mon May  4 16:57:06 2020

@author: oislen
"""



import file_constants as cons
import pandas as pd
import utilities as utl

agg_base = pd.read_feather(cons.base_agg_comp_fpath)

agg_base['item_id'].value_counts()
agg_base['shop_id'].value_counts()
agg_base['item_cnt_day'].value_counts()[0:20]
(agg_base['item_id'].astype(str) + '_' + agg_base['shop_id'].astype(str)).value_counts()

group_cols = ['item_id']
agg_base.groupby(group_cols).agg()
agg_base['date_block_num'].unique()

dbn = 1

dbn_chunks = agg_base[agg_base['date_block_num'] == dbn]
dbn_chunks['item_id'].value_counts()
dbn_chunks.isnull().sum()

train = agg_base[agg_base['data_split'] == 'train']
valid = agg_base[agg_base['data_split'] == 'valid']
test = agg_base[agg_base['data_split'] == 'test']
holdout = agg_base[agg_base['data_split'] == 'holdout']


holdout['item_price'].value_counts(dropna = False)
(holdout['item_price'] == -999).sum()
