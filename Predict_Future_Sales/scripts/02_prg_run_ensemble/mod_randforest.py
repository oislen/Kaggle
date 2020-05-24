# -*- coding: utf-8 -*-
"""
Created on Mon May 18 20:42:41 2020

@author: oislen
"""

from sklearn.ensemble import RandomForestRegressor
from meta_level_I.exe_model import exe_model
import numpy as np

def mod_randforest(cons, max_dept, rand_state, feat_imp, n, date, skip_train, model_type):
    
    """
    """
    
    # set model pk output file path
    model_name = '{}_dept{}'.format(model_type, max_dept)
    model_pk_fpath = '{}/{}_model.joblib'.format(cons.models_dir, model_name)
    cv_sum_fpath = '{}/{}_cv_summary.csv'.format(cons.cv_results_dir, model_name)
    
    print(model_pk_fpath)
    print(cv_sum_fpath)
    
    # initiate random forest model
    model = RandomForestRegressor()
    
    print(model)
    
    # set model parameters
    model_params = {'criterion':['mse'],
                    'max_depth':[max_dept],
                    'random_state':[rand_state],
                    'n_estimators':[25],
                    'max_features':[np.int8(np.floor(n / i)) for i in [1, 2]]
                    }
    
    print(model_params)
    
    # set the train, valid and test sub limits
    #train_cv_split_dict = [{'train_sub':idx, 'valid_sub':idx + 1} for idx in np.arange(start = 12, stop = 29, step = 13)]
    train_cv_split_dict = [{'train_sub':28, 'valid_sub':29}]
        
    # set the train, valid and test sub limits
    test_split_dict = {'train_sub':29, 'valid_sub':32, 'test_sub':33}
    
    # set predictions file path
    mod_preds = '{}/{}_{}'.format(cons.pred_data_dir, model_name, date)

    print(mod_preds)
    
    # set the input data file path
    data_fpath = cons.model_data_fpath
    
    print(data_fpath)
    
    # execute the model
    exe_model(cons = cons,
              feat_imp = feat_imp,
              data_fpath = data_fpath,
              model = model,
              model_params = model_params,
              train_cv_split_dict = train_cv_split_dict,
              model_pk_fpath = model_pk_fpath,
              mod_preds = mod_preds,
              cv_sum_fpath = cv_sum_fpath,
              test_split_dict = test_split_dict,
              n = n,
              model_name = model_name,
              skip_train = skip_train
              )
    
    return