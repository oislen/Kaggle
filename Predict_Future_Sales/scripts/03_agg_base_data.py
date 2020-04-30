# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 20:33:00 2020

@author: oislen
"""

import pandas as pd
import file_constants as cons
import clean_constants as clean_cons

def agg_base_data():

    """
    """
    
    print('loading base data ...')
    
    # load in base data
    base_raw = pd.read_feather(cons.base_raw_data_fpath)
    
    print('aggregating base data ...')
    
    # want to aggregate to year, month, shop and product level
    agg_base = base_raw.groupby(clean_cons.group_cols, as_index = False).agg(clean_cons.agg_dict)
    
    print('outputting aggregated base data ...')
    
    # output aggreated base data as feather file
    agg_base.to_feather(cons.base_agg_data_fpath)
    
    return

agg_base_data()