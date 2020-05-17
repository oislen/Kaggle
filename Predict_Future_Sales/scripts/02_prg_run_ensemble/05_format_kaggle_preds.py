# -*- coding: utf-8 -*-
"""
Created on Sun May 17 12:29:39 2020

@author: oislen
"""

import pandas as pd

def format_kaggle_preds(cons):
    
    """
    """
    
    y_holdout = pd.read_feather(cons)
    
    # TODO: possibly extract this last step out into a seperate script
    # map items with no historical sell to 0
    y_holdout.columns
    no_sales_hist_filt = y_holdout['no_sales_hist_ind'] == 1
    y_holdout.loc[no_sales_hist_filt, ['y_holdout_pred']] = 0
    
    # extract out test predictions
    holdout_subset_filt = y_holdout['holdout_subset_ind'] == 1
    holdout_out = y_holdout.loc[holdout_subset_filt, ['ID', 'y_holdout_pred']]
    holdout_out = holdout_out.rename(columns = {'y_holdout_pred':'item_cnt_month'})
    holdout_out_sort = holdout_out.sort_values(by = ['ID']).astype(int)
    
    # output predictions as csv file
    output_foath = cons.pred_data_dir + '/randforest20200516.csv'
    holdout_out_sort.to_csv(output_foath,
                            index = False
                            )