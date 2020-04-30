# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 20:52:18 2020

@author: oislen
"""

import pandas as pd
import file_constants as cons
import clean_constants as clean_cons

pd.set_option('display.max_rows', 30)
pd.set_option('display.max_columns', 30)

base_raw = pd.read_feather(cons.base_raw_data_fpath)

base_raw.columns
base_raw.head()
base_raw.dtypes

# want to aggregate to year, month, shop and product level


agg_base = base_raw.groupby(clean_cons.group_cols, as_index = False).agg(clean_cons.agg_dict)

#
agg_base