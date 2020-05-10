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
base['holdout_subset_ind'].value_counts(dropna = False)

pd.crosstab(index = base['no_sales_hist_ind'], columns = base['item_cnt_day'].isin([0, -999]).astype(int))
pd.crosstab(index = base['no_sales_hist_ind'], columns = base['holdout_subset_ind'])

# Step 1:
# TODO: encode categorical variables; alphabetical / order encode
# TODO: create shift attribues; last month, last three months, last year
# TODO: create date window attributes; quarters, seasons
# TODO: create mean encoded attributes; item category, item_id, shop_id
# TODO: create interaction attributes

# Step 2:
# TODO: split into train, valid, test and holdout
# TODO: fit random forest and gradient boosting decision trees; loss optimise on MSE
# TODO: plot early stopping plot
# TODO: tune hyperparameters
# TODO: predict any item with no sales history as 0
