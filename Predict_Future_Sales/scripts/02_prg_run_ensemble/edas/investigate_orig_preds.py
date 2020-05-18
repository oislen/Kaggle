# -*- coding: utf-8 -*-
"""
Created on Mon May 18 09:07:58 2020

@author: oislen
"""


import pandas as pd
import seaborn as sns

preds1 = pd.read_csv('C:/Users/User/Documents/GitHub/Kaggle/Predict_Future_Sales/data/pred/randforest20200514.csv')
preds2 = pd.read_csv('C:/Users/User/Documents/GitHub/Kaggle/Predict_Future_Sales/data/pred/randforest20200518.csv')

preds_join = pd.merge(left = preds1, right = preds2, on = 'ID', how = 'outer', suffixes = ('_orig', '_new')) 

preds_join['item_cnt_month_orig'].value_counts()
preds_join['item_cnt_month_new'].value_counts()

sns.scatterplot(x = 'item_cnt_month_orig', y = 'item_cnt_month_new', data = preds_join)
