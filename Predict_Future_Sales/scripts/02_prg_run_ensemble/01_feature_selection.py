# -*- coding: utf-8 -*-
"""
Created on Sun May 17 15:59:25 2020

@author: oislen
"""

import pandas as pd
import numpy as np
import utilities_ensemble as utl_ens
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

pd.set_option('display.max_columns', 10)

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

    # TODO: convert this into a function
    data_cols = train_data.columns
    data_dtypes = train_data.dtypes
    data_dtypes.value_counts()
    
    int32_cols = data_cols[data_dtypes == np.int32]
    int64_cols = data_cols[data_dtypes == np.int64]
    float32_cols = data_cols[data_dtypes == np.float64]
    
    train_data[int32_cols] = train_data[int32_cols].astype(np.int8)
    train_data[int64_cols] = train_data[int64_cols].astype(np.int8)
    train_data[float32_cols] = train_data[float32_cols].astype(np.float32)

    train_data.dtypes.value_counts()

    #-- Random Forest Feature Importance --#
    
    print('Running random forest feature importance ...')
    
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
    feat_imp_fpath = cons.randforest_feat_imp
    rf_feat_imp = utl_ens.feat_imp_sum(model = rfc, 
                                       pred_cols = pred_cols, 
                                       feat_imp_fpath = feat_imp_fpath
                                       )
    
    print(rf_feat_imp.head(20))
    
    #-- LASSO Feature Importance --#
    
    #-- Gradient Boosting Feature Importance --#
    
    print('Running gradient boosting feature importance ...')
    
    # initiate random forest model
    gbr = GradientBoostingRegressor(max_depth = 3, 
                                    random_state = 1234, 
                                    criterion = 'mse',
                                    n_estimators = 100,
                                    verbose = 2,
                                    max_features = 'auto'
                                    )
    
    # fit random forests model
    gbr.fit(train_data[pred_cols], train_data['item_cnt_day'])
    
    # extract feature importance
    feat_imp_fpath = cons.gradboost_feat_imp
    gb_feat_imp = utl_ens.feat_imp_sum(model = gbr, 
                                       pred_cols = pred_cols, 
                                       feat_imp_fpath = feat_imp_fpath
                                       )
    
    print(gb_feat_imp.head(20))