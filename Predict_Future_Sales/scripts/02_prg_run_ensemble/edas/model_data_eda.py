# -*- coding: utf-8 -*-
"""
Created on Fri May 29 09:30:04 2020

@author: oislen
"""


import pandas as pd
import os
import sys 
import seaborn as sns

# add requried reference file path to paths
cwd = os.getcwd()
pwd = cwd.split('\\')
ref_dir = os.path.join('\\'.join(pwd[:-1]), 'reference')
par_dir = '\\'.join(pwd[:-2])
for path in [ref_dir, par_dir]:
    sys.path.append(path)

import file_constants as cons
#import clean_utilities as utl
import numpy as np

model_data = pd.read_feather(cons.model_data_fpath)

model_data.shape

model_data.columns
model_data.dtypes
model_data.dtypes.value_counts()

# check no 0 item price
# check 
# check unique shop item combination is 214200

model_data['shop_item_id'] = model_data['shop_id'].astype(str) + '_' + model_data['item_id'].astype(str)
model_data['shop_item_id'].nunique() == 214200


model_data['primary_key'].value_counts()
model_data['item_price'].value_counts()
model_data['item_cnt_day'].value_counts()




# first value should be positive integer
agg_shop_item_date_tab = model_data.groupby(['date_block_num', 'shop_id', 'item_id']).agg({'item_price':'first'})
agg_shop_item_date_tab = model_data.groupby(['shop_id', 'item_id', 'date_block_num']).agg({'item_price':'first'})
agg_shop_item_tab = model_data.groupby(['shop_id', 'item_id']).agg({'item_price':'first'})

model_data.groupby(['date_block_num', 'shop_id', 'item_id']).agg({'item_price':'first'})




hist_tab = pd.pivot_table(data = model_data,
                          index = ['date_block_num'],
                          columns = ['shop_id', 'item_id'],
                          values = 'no_sales_hist_ind',
                          aggfunc = np.sum,
                          dropna = True
                          )

hist_tab = (hist_tab == 1).astype(int)







price_tab = pd.pivot_table(data = model_data,
                           index = ['date_block_num'],
                           columns = ['shop_id', 'item_id'],
                           values = 'item_price',
                           aggfunc = np.sum,
                           dropna = True
                           )

zero_prices = (price_tab == 0).astype(int)


check = hist_tab == zero_prices
check.all().all()


item_tab = pd.pivot_table(data = model_data,
                          index = ['date_block_num'],
                          columns = ['shop_id', 'item_id'],
                          values = 'item_cnt_day',
                          aggfunc = np.sum,
                          dropna = True
                          )

sold_itms = (item_tab >= 1).astype(int)


zero_price_sold = zero_prices & sold_itms


(zero_price_sold == 1).any().any()
