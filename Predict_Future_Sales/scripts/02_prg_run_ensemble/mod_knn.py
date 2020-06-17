# -*- coding: utf-8 -*-
"""
Created on Sun May 31 19:54:33 2020

@author: oislen
"""

from sklearn.neighbors import KNeighborsRegressor
from meta_level_I.exe_model import exe_model

def mod_knn(cons, feat_imp, n, date, skip_train, model_type):
    
    """
    """
    
    # set model pk output file path
    model_name = '{}'.format(model_type)
    model_pk_fpath = '{}/{}_model.pkl'.format(cons.models_dir, model_name)
    cv_sum_fpath = '{}/{}_cv_summary.csv'.format(cons.cv_results_dir, model_name)
    
    print(model_pk_fpath)
    print(cv_sum_fpath)
    
    # initiate decision tree regressor
    model = KNeighborsRegressor()
    
    print(model)
    
    # set model parameters
    model_params = {'n_neighbors':[3, 4, 5, 6, 7],
                    'weights':['uniform', 'distance'],
                    'algorithm':['auto'],
                    'p':[1, 2, 3, 4]
                    }
    
    print(model_params)
    
    # set the train, valid and test sub limits
    #train_cv_split_dict = [{'train_sub':idx, 'valid_sub':idx + 1} for idx in np.arange(start = 12, stop = 29, step = 5)]
    train_cv_split_dict = [{'train_sub':28, 'valid_sub':29}]
    
    # set the train, valid and test sub limits
    test_split_dict = {'train_sub':29, 'valid_sub':32, 'test_sub':33}
    
    # set predictions
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