# -*- coding: utf-8 -*-
"""
Created on Sun May 17 15:59:51 2020

@author: oislen
"""

import pandas as pd
import utilities_ensemble as utl_ens
from sklearn.model_selection import GridSearchCV
import joblib as jl
import pickle as pk

pd.set_option('display.max_columns', 10)

def cv_model_train(data_fpath,
                   tar_cols,
                   pred_cols,
                   model,
                   model_params,
                   train_cv_split_dict,
                   model_pk_fpath,
                   cv_sum_fpath 
                   ):
    
    """
    """
    
    print('loading base data {} ...'.format(data_fpath))
    
    # load in model data
    base = pd.read_feather(data_fpath)

    print('creating cv index splits ...')
    
    # generate indices for cv splits
    cv_list = utl_ens.gen_cv_splits(dataset = base,
                                    train_cv_split_dict = train_cv_split_dict
                                    )
    
    # set the refit boolean
    refit_bool = True
    
    print('creating grid search object ...')
    
    # initiate CV object
    gcv = GridSearchCV(estimator = model, 
                       param_grid = model_params,
                       scoring = 'neg_root_mean_squared_error',
                       n_jobs = 2,
                       cv = cv_list,
                       refit = refit_bool,
                       verbose = 2
                       )
    
    print('running grid search cv ...')
    
    print(pred_cols)
    
    # Split the predictors from the target
    sub_index = [val for lst in cv_list[-1] for val in lst]
    X = base.loc[sub_index, pred_cols]
    y = base.loc[sub_index, tar_cols[0]]
    
    print(X.head())
    print(y.head())
    
    # fit cv object
    gcv.fit(X, y)
    
    print('generating summary ...')
    
    # generate a summary of the cv results
    cv_results = utl_ens.gen_cv_sum(gcv, cv_sum_fpath)
    
    print(cv_results.head())
    
    print('outputting best model {} ...'.format(model_pk_fpath))
    
    # refit model
    bdtr = gcv.best_estimator_
    #bdtr = bdtr.fit(X, y)
    # pickle best estimator
    jl.dump(bdtr, model_pk_fpath)
    pk.dump(bdtr, open(model_pk_fpath, "wb"), protocol = pk.HIGHEST_PROTOCOL)
    
    return