# -*- coding: utf-8 -*-
"""
Created on Sun May  3 15:55:58 2020

@author: oislen
"""

import pandas as pd
import os 
import sys

# add requried reference file path to paths
cwd = os.getcwd()
pwd = cwd.split('\\')
ref_dir = os.path.join('\\'.join(pwd[:-1]), 'reference')
par_dir = '\\'.join(pwd[:-2])
for path in [ref_dir, par_dir]:
    sys.path.append(path)

import file_constants as cons
import clean_utilities as utl

agg_base = pd.read_feather(cons.base_agg_data_fpath)


agg_base.shape

agg_base.columns

agg_base['shop_item_id'] = agg_base['item_id'].astype(str) + '_' + agg_base['shop_id'].astype(str)
agg_base['shop_item_id'].nunique()

shift_data = utl.gen_shift_attr(dataset = agg_base,
                                values = ['item_cnt_day'],
                                index = ['date_block_num'],
                                columns = ['shop_id', 'item_id']
                                )


agg_base.columns
agg_base.groupby(['shop_id', 'item_id', 'date_block_num']).agg({'item_cnt_day':'sum'})
