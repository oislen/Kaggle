# -*- coding: utf-8 -*-
"""
Created on Sun May 17 12:29:39 2020

@author: oislen
"""

import pandas as pd
import numpy as np

def format_kaggle_preds(pred_paths,
                        kaggle_preds
                        ):
    
    """
    """
    
    # extract out the prediciton paths
    y_valid_preds_path = pred_paths['y_valid_preds_path']
    y_test_preds_path = pred_paths['y_test_preds_path']
    y_holdout_preds_path = pred_paths['y_holdout_preds_path']
    
    y_holdout = pd.read_csv(y_holdout_preds_path)
    
    # TODO: possibly extract this last step out into a seperate script
    # TODO: move this mapping and flooring into model predictions script
    # map items with no historical sell to 0
    y_holdout.columns
    no_sales_hist_filt = y_holdout['no_sales_hist_ind'] == 1
    y_holdout.loc[no_sales_hist_filt, ['y_holdout_pred']] = 0
    
    # round down remaining results to nearest value
    y_holdout['y_holdout_pred'] = y_holdout['y_holdout_pred'].apply(lambda x: np.floor(x))
    y_holdout['y_holdout_pred'].value_counts()
    
    # extract out test predictions
    holdout_subset_filt = y_holdout['holdout_subset_ind'] == 1
    holdout_out = y_holdout.loc[holdout_subset_filt, ['ID', 'y_holdout_pred']]
    holdout_out = holdout_out.rename(columns = {'y_holdout_pred':'item_cnt_month'})
    holdout_out_sort = holdout_out.sort_values(by = ['ID']).astype(int)
    
    # output predictions as csv file
    holdout_out_sort.to_csv(kaggle_preds,
                            index = False
                            )
    
    return