# -*- coding: utf-8 -*-
"""
Created on Sun May 17 10:59:45 2020

@author: oislen
"""

import pandas as pd
import numpy as np

def extract_model_cols(dataset):
    
    """
    """
    
    print('extracting out dataset columns ...')
    
    # seperate predictors from response
    data_cols = dataset.columns.tolist()
    index_cols = ['primary_key', 'ID', 'data_split', 'no_sales_hist_ind', 'holdout_subset_ind']
    tar_cols = ['item_cnt_day']
    excl_cols = ['item_category_id', 'item_cat', 'item_cat_sub', 
                 'shop_id_total_item_cnt_day', 'item_id_total_item_cnt_day',
                 'shop_item_id', 'revenue', 'item_price', 'no_holdout_sales_hist_ind'
                 ]
    pred_cols = [col for col in data_cols if col not in index_cols + tar_cols + excl_cols]
    
    # create a dictionary of the output columns
    model_cols_dict = {'index_cols':index_cols,
                       'tar_cols':tar_cols,
                       'excl_cols':excl_cols,
                       'pred_cols':pred_cols
                       }
    
    return model_cols_dict
    

def gen_cv_splits(dataset,
                  cv_split_dict
                  ):
    
    """
    """
    
    cv_list = []
    
    for splits_limits in cv_split_dict:
        
        print(splits_limits)
        print('taking subset of data ...')
        
        sub_cols = ['date_block_num', 'data_split', 'primary_key']
        data = dataset[sub_cols]
        
        print('creating data split sub column ...')
        
        # extract out the train, valid and test subset limits
        trn_sl = splits_limits['train_sub']
        vld_sl = splits_limits['valid_sub']
        
        print('extracting data splits ...')
    
        # define data split filters
        filt_train = data['date_block_num'] <= trn_sl
        filt_valid = data['date_block_num'] == vld_sl
            
        # extract out the data splits
        train_data = data[filt_train].index.values.astype(int)
        valid_data = data[filt_valid].index.values.astype(int)
    
        print('creating output dictionary ...')
    
        # create a dictionary of data splits
        data_splits_dict = {'train_data_idx':train_data,
                            'valid_data_idx':valid_data
                            }

        cv_list.append((data_splits_dict['train_data_idx'], data_splits_dict['valid_data_idx']))
    

    return cv_list



def gscv_sum(clf):
    
    """
    """
    
    cv_results = pd.DataFrame({'params':clf.cv_results_['params'],
                               'mean_test_score':clf.cv_results_['mean_test_score'] * -1,
                               'std_test_score':clf.cv_results_['std_test_score'],
                               'rank_test_score':clf.cv_results_['rank_test_score']
                               })
    
    cv_results = cv_results.sort_values(by = ['rank_test_score'])
    
    return cv_results
    


def extract_data_splits(dataset,
                        index_cols,
                        tar_cols,
                        pred_cols,
                        data_splits_limits
                        ):
    
    """
    """

    print('generating data splits ...')
    
    print('creating data split sub column ...')
    
    # extract out the train, valid and test subset limits
    trn_sl = data_splits_limits['train_sub']
    vld_sl = data_splits_limits['valid_sub']
    tst_sl = data_splits_limits['test_sub']
    

    print('extracting data splits ...')
    
    # define data split filters
    filt_train = dataset['date_block_num'] <= trn_sl
    filt_valid = dataset['date_block_num'] == vld_sl
    filt_test = dataset['date_block_num'] == tst_sl
    filt_holdout = dataset['date_block_num'] == 34
        
    # extract out the data splits
    train_data = dataset[filt_train]
    valid_data = dataset[filt_valid]
    test_data = dataset[filt_test]
    holdout_data = dataset[filt_holdout]

    # split datasets into train, valid, test and holdout
    X_train = train_data[index_cols + pred_cols]
    y_train = train_data[index_cols + tar_cols]
    X_valid = valid_data[index_cols + pred_cols]
    y_valid = valid_data[index_cols + tar_cols]
    X_test = test_data[index_cols + pred_cols]
    y_test = test_data[index_cols + tar_cols]
    X_holdout = holdout_data[index_cols + pred_cols]
    y_holdout = holdout_data[index_cols + tar_cols]
    
    print('creating output dictionary ...')
    
    # create a dictionary of data splits
    data_splits_dict = {'train_data':train_data,
                        'valid_data':valid_data,
                        'test_data':test_data,
                        'holdout_data':holdout_data,
                        'X_train':X_train,
                        'y_train':y_train,
                        'X_valid':X_valid,
                        'y_valid':y_valid,
                        'X_test':X_test,
                        'y_test':y_test,
                        'X_holdout':X_holdout,
                        'y_holdout':y_holdout
                        }


    
    return data_splits_dict


def feat_imp_sum(model, 
                 pred_cols, 
                 feat_imp_fpath
                 ):
           
    # extract feature importance
    feat_imp = pd.DataFrame({'attr':pred_cols,
                             'feat_imp':model.feature_importances_ * 100
                             })
    
    # sort by importance
    feat_imp = feat_imp.sort_values(by = 'feat_imp', ascending = False)
    
    # reset the index
    feat_imp = feat_imp.reset_index(drop = True)
    
    # output to .csv file
    feat_imp.to_csv(feat_imp_fpath, index = False)
    
    return feat_imp
    
def extract_feat_imp(cons, 
                     model_type, 
                     n = 20
                     ):
    
    """
    """
    
    index_cols = ['primary_key', 'ID', 'data_split', 'holdout_subset_ind', 'no_sales_hist_ind']
    tar_cols = ['item_cnt_day']
    req_cols = ['year', 'month', 'date_block_num', 'item_id', 'shop_id']
    
    if model_type == 'randforest':
        
        feat_imp = pd.read_csv(cons.randforest_feat_imp)
    
    elif model_type == 'gradboost':
        
        feat_imp = pd.read_csv(cons.gradboost_feat_imp)
    
    feat_imp_cols = feat_imp['attr'].head(n).tolist()
    
    pred_cols = list(set(feat_imp_cols + req_cols))
    
    # create a dictionary of the output columns
    model_cols_dict = {'index_cols':index_cols,
                       'tar_cols':tar_cols,
                       'pred_cols':pred_cols
                       }
    
    return model_cols_dict



def format_preds(dataset, preds_cols):
    
    """
    """
    
    data = dataset.copy(True)
    
    # map items with no historical sell to 0
    no_sales_hist_filt = data['no_sales_hist_ind'] == 1
    data.loc[no_sales_hist_filt, [preds_cols]] = 0
    
    # round down remaining results to nearest value
    data[preds_cols] = data[preds_cols].apply(lambda x: np.floor(x))
    print(data[preds_cols].value_counts())
    
    return data
