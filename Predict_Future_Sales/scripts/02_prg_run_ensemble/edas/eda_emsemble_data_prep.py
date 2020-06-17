# -*- coding: utf-8 -*-
"""
Created on Sun May 17 12:17:57 2020

@author: oislen
"""
 

# read in the base data
feather_file = 'C:/Users/User/Documents/GitHub/Kaggle/Predict_Future_Sales/data/base/base_supp_data.feather'
# feather_file = cons.base_agg_supp_fpath


base = pd.read_feather(feather_file)

base.columns

# TODO move to eda script
base['item_price'].value_counts(dropna = False)
base['item_cnt_day'].value_counts(dropna = False)
#base['n_refund'].value_counts(dropna = False)
#base['n_sale'].value_counts(dropna = False)
base['data_split'].value_counts(dropna = False)
base['ID'].value_counts(dropna = False)
base['holdout_subset_ind'].value_counts(dropna = False)

pd.crosstab(index = base['no_sales_hist_ind'], columns = base['item_cnt_day'].isin([0, -999]).astype(int))
pd.crosstab(index = base['no_sales_hist_ind'], columns = base['holdout_subset_ind'])
