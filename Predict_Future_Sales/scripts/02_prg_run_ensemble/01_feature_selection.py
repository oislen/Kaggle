# -*- coding: utf-8 -*-
"""
Created on Sun May 17 15:59:25 2020

@author: oislen
"""

import pandas as pd
import utilities_ensemble as utl_ens
from sklearn.ensemble import RandomForestRegressor

def gen_feature_selection(cons):
    
    """
    """
        
    # load in model data
    base = pd.read_feather(cons.model_data_fpath)
    
    # seperate predictors from response
    model_cols_dict = utl_ens.extract_model_cols(base)
    index_cols = model_cols_dict['index_cols']
    tar_cols = model_cols_dict['tar_cols']
    pred_cols = model_cols_dict['pred_cols']
    
    # filter out the training data
    filt_train_data = base['data_split'] == 'train'
    train_data = base[filt_train_data]
    
    #-- Random Forest Feature Importance --#
    
    # initiate random forest model
    rfc = RandomForestRegressor(max_depth = 7, 
                                random_state = 1234, 
                                criterion = 'mse',
                                n_estimators = 100,
                                n_jobs = 2,
                                verbose = 2,
                                max_features = 'auto'
                                )
    
    # fit random forests model
    rfc.fit(train_data[pred_cols], train_data['item_cnt_day'])
    
    # extract feature importance
    feat_imp_fpath = '{}/randforest_feat_imp.csv'.format(cons.feat_imp_dir)
    rf_feat_imp = utl_ens.feat_imp_sum(model = rfc, pred_cols = pred_cols, feat_imp_fpath = feat_imp_fpath)
    
    print(rf_feat_imp.head(10))
    
    #-- LASSO Feature Importance --#
    
    #-- Gradient Boosting Feature Importance --#
    