# -*- coding: utf-8 -*-
"""
Created on Sun May 10 19:15:38 2020

@author: oislen
"""

import pandas as pd

feather_file = 'C:/Users/User/Documents/GitHub/Kaggle/Predict_Future_Sales/data/base/base_supp_data.feather'

base = pd.read_feather(feather_file)

base.columns

base['item_price'].value_counts(dropna = False)
base['item_cnt_day'].value_counts(dropna = False)
base['n_refund'].value_counts(dropna = False)
base['n_sale'].value_counts(dropna = False)
base['data_split'].value_counts(dropna = False)
base['ID'].value_counts(dropna = False)
(base.loc[base['data_split'] == 'holdout', 'ID'] != -999).sum()



