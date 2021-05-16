# -*- coding: utf-8 -*-
"""
Created on Sun May 17 15:59:51 2020

@author: oislen
"""

import cons
import pandas as pd
from sklearn.model_selection import GridSearchCV
import joblib as jl
import pickle as pk
from reference.gen_cv_splits import gen_cv_splits
from reference.gen_cv_sum import gen_cv_sum

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
    
    Cross- Validation Model Training Documentation
    
    Function Overview
    
    This performs a grid search cross-validation to find the optimal combination of model parameters and train the model.
    The optimal model is output after the hyper-parameter tuning is completed.
    
    Defaults
    
    cv_model_train(data_fpath,
                   tar_cols,
                   pred_cols,
                   model,
                   model_params,
                   train_cv_split_dict,
                   model_pk_fpath,
                   cv_sum_fpath
                   )
    
    Parameters
    
    data_fpath - String, the full file path to the training data
    tar_cols - List of Strings, the target columns within the training data
    pred_cols - List of Strings, the predictor columns within the training data
    model - Sklearn Model, the sci-kit learn model to train
    model_params - Dictionary, the model parameters to tune using grid search cross-validation
    train_cv_split_dict - Dictionary, the splitting configurations for the cross-validation training
    model_pk_fpath - String, the full file path to output the model as a .pkl file
    cv_sum_fpath - String, the full file path to output the cross-validation summary
    
    Returns
    
    0 for successful execution
    
    Example
    
    model_train.cv_model_train(data_fpath = data_fpath,
                               tar_cols = tar_cols,
                               pred_cols = pred_cols,
                               model = DecisionTreeRegressor(),
                               model_params = {'criterion':['mse', 'friedman_mse']},
                               train_cv_split_dict = {'train_sub':29, 'valid_sub':32, 'test_sub':33},
                               model_pk_fpath = model_pk_fpath,
                               cv_sum_fpath = cv_sum_fpath
                               )

    
    """
    
    print('loading base data {} ...'.format(data_fpath))
    
    # load in model data
    base = pd.read_feather(data_fpath)

    print('creating cv index splits ...')
    
    # generate indices for cv splits
    cv_list = gen_cv_splits(dataset = base,
                            train_cv_split_dict = train_cv_split_dict
                            )
    
    print('creating grid search object ...')
    
    # initiate CV object
    gcv = GridSearchCV(estimator = model, 
                       param_grid = model_params,
                       scoring = 'neg_root_mean_squared_error',
                       n_jobs = cons.n_cpu,
                       cv = cv_list,
                       refit = cons.refit_bool,
                       verbose = cons.verbose
                       )
    
    print(model_params)
    
    print('N Jobs: {n_cpu}'.format(n_cpu = cons.n_cpu))
    
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
    cv_results = gen_cv_sum(gcv, cv_sum_fpath)
    
    print(cv_results.head())
    
    print('outputting best model {} ...'.format(model_pk_fpath))
    
    # refit model
    bdtr = gcv.best_estimator_
    #bdtr = bdtr.fit(X, y)
    # pickle best estimator
    jl.dump(bdtr, model_pk_fpath)
    pk.dump(bdtr, open(model_pk_fpath, "wb"), protocol = pk.HIGHEST_PROTOCOL)
    
    return 0