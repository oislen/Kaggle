# -*- coding: utf-8 -*-
"""
Created on Sun May  3 15:55:58 2020

@author: oislen
"""

import file_constants as cons
import pandas as pd
import utilities as utl

agg_base = pd.read_feather(cons.base_agg_data_fpath)

shift_data = utl.gen_shift_attr(dataset = agg_base,
                                values = ['item_cnt_day'],
                                index = ['date_block_num'],
                                columns = ['shop_id', 'item_id']
                                )