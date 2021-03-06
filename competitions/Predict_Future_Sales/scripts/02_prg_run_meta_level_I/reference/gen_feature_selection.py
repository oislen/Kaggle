# -*- coding: utf-8 -*-
"""
Created on Sun May 17 15:59:25 2020

@author: oislen
"""

import cons
import pandas as pd
from reference.extract_model_cols import extract_model_cols
from reference.feat_imp_sum import feat_imp_sum
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

def gen_feature_selection(feat_type = 'randforest'):
    
    """
    
    Generate Feature Selction Documentation
    
    Function Overview
    
    This function does a high level feature importance analysis using random forest and gradient boosting trees.
    The goal is just get a broad understanding of which features are the most predictive of monthly item sales.
    The results will feed into the next stage where models are fitted.
    
    Defaults
    
    gen_feature_selection(feat_type = 'randforest')
    
    Parameters
    
    feat_type - String, the type of model to use for generating the feature importance, default is 'randforest'
    
    Returns
    
    0 for successful execution
    
    Example
    
    gen_feature_selection(feat_type = 'randforest')
    
    """
        
    # load in model data
    base = pd.read_feather(cons.model_data_fpath)
    
    # seperate predictors from response
    model_cols_dict = extract_model_cols(base)
    pred_cols = model_cols_dict['pred_cols']
    
    # filter out the training data
    filt_train_data = base['meta_level'] == 'level_1'
    train_data = base[filt_train_data]

    ########################################
    #-- Random Forest Feature Importance --#
    ########################################
    
    if feat_type == 'randforest':
    
        print('Running random forest feature importance ...')
        
        # initiate random forest model
        rfc = RandomForestRegressor(max_depth = cons.feat_imp_max_depth, 
                                    random_state = cons.rand_seed, 
                                    criterion = cons.feat_imp_criterion,
                                    n_estimators = cons.feat_imp_n_estimators,
                                    n_jobs = cons.n_cpu,
                                    verbose = cons.verbose,
                                    max_features = cons.feat_imp_max_features
                                    )
        
        # fit random forests model
        rfc.fit(train_data[pred_cols], train_data['item_cnt_day'])
        
        # extract feature importance
        feat_imp_fpath = cons.randforest_feat_imp
        rf_feat_imp = feat_imp_sum(model = rfc, 
                                   pred_cols = pred_cols, 
                                   feat_imp_fpath = feat_imp_fpath
                                   )
        
        print(rf_feat_imp.head(20))
        
    ################################
    #-- LASSO Feature Importance --#
    ################################
    
    ############################################
    #-- Gradient Boosting Feature Importance --#
    ############################################
    
    elif feat_type == 'gradboost':
        
        print('Running gradient boosting feature importance ...')
        
        # initiate random forest model
        gbr = GradientBoostingRegressor(max_depth = cons.feat_imp_max_depth, 
                                        random_state = cons.rand_seed, 
                                        criterion = cons.feat_imp_criterion,
                                        n_estimators = cons.feat_imp_n_estimators,
                                        verbose = cons.verbose,
                                        #validation_fraction = 0.1,
                                        #n_iter_no_change = 5,
                                        max_features = cons.feat_imp_max_features
                                        )
        
        # fit random forests model
        gbr.fit(train_data[pred_cols], train_data['item_cnt_day'])
        
        # extract feature importance
        feat_imp_fpath = cons.gradboost_feat_imp
        gb_feat_imp = feat_imp_sum(model = gbr, 
                                   pred_cols = pred_cols, 
                                   feat_imp_fpath = feat_imp_fpath
                                   )
        
        print(gb_feat_imp.head(20))
        
    return 0