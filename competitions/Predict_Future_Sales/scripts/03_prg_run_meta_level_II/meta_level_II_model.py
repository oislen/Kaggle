# -*- coding: utf-8 -*-
"""
Created on Sat May 22 10:40:08 2021

@author: oislen
"""
import cons
import pandas as pd
import statsmodels.api as sm
from plot_preds_hist import plot_preds_hist

def meta_level_II_model(join_data, validation = True):
    
    """
    
    Meta-Level II Model Documentation
    
    Function Overview
    
    This function trains, fits and predicts the meta-level II model
    
    Defaults
    
    meta_level_II_model(join_data)
    
    Parameters
    
    join_data - DataFrame, the meta-level I features
    
    Returns
    
    0 for successful execution
    
    Example
    
    meta_level_II_model(join_data = join_data)
    
    """
        
    ##################
    #-- Split Data --#
    ##################
    
    if validation == True:
        train_date_block = [30, 31]
        valid_date_block = [32]
        test_date_block = [33]
        holdout_data_block = [34]
    else:
        train_date_block = [30, 31, 32, 33]
        valid_date_block = []
        test_date_block = []
        holdout_data_block = [34]
    
    # create filter conditions to split out data
    filt_train = join_data['date_block_num'].isin(train_date_block)
    filt_valid = join_data['date_block_num'].isin(valid_date_block)
    filt_test = join_data['date_block_num'].isin(test_date_block)
    filt_holdout = join_data['date_block_num'].isin(holdout_data_block)
    
    # split into train, valid, test and holdout
    train = join_data[filt_train]
    valid = join_data[filt_valid]
    test = join_data[filt_test]
    holdout = join_data[filt_holdout]
    
    # extract out columns
    meta_cols = join_data.columns
    
    # generate the relevant groups of columns / attributes
    tar_col = cons.meta_level_II_resp_col
    index_cols = [col for col in cons.meta_level_II_base_cols if col not in tar_col]
    pred_cols = [col for col in meta_cols if col not in tar_col + index_cols]
    
    # split up train, valid, test and holdout into X and y
    X_train = train[index_cols + pred_cols]
    y_train = train[index_cols + tar_col]
    X_valid = valid[index_cols + pred_cols]
    y_valid = valid[index_cols + tar_col]
    X_test = test[index_cols + pred_cols]
    y_test = test[index_cols + tar_col]
    X_holdout = holdout[index_cols + pred_cols]
    y_holdout = holdout[index_cols + tar_col]
    
    #############################
    #-- Create Level II Model --#
    #############################
    
    # poisson model
    pois_fam = sm.families.Poisson()
    model = sm.GLM(endog = y_train['item_cnt_day'], exog = X_train[pred_cols], family = pois_fam)
    model = model.fit()
    model.params
    model.summary()
    
    # make predictions
    y_valid['meta_lvl_II_preds'] = model.predict(X_valid[pred_cols]).clip(cons.lower_bound, cons.upper_bound)
    y_test['meta_lvl_II_preds'] = model.predict(X_test[pred_cols]).clip(cons.lower_bound, cons.upper_bound)
    y_holdout['meta_lvl_II_preds'] = model.predict(X_holdout[pred_cols]).clip(cons.lower_bound, cons.upper_bound)

    # plot histgrams
    plot_preds_hist(dataset = y_valid, pred = 'item_cnt_day', model_name = 'poisson')
    plot_preds_hist(dataset = y_valid, pred = 'meta_lvl_II_preds', model_name = 'poisson')
    plot_preds_hist(dataset = y_test, pred = 'meta_lvl_II_preds', model_name = 'poisson')
    plot_preds_hist(dataset = y_holdout, pred = 'meta_lvl_II_preds', model_name = 'poisson')
    
    # create the output results dictionary
    sub_dict = {"ID": y_holdout['ID'].astype(int), 
                "item_cnt_month": y_holdout['meta_lvl_II_preds'].rename({'meta_lvl_II_preds':'item_cnt_month'})
                }
    
    # convert dictionary to dataframe
    submission = pd.DataFrame(sub_dict).sort_values(by = ['ID'])
    
    # write output predictions to a .csv file
    submission.to_csv(cons.meta_level_II_preds_fpath, index = False)
    
    return 0