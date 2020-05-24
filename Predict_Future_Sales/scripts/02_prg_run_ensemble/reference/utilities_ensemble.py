# -*- coding: utf-8 -*-
"""
Created on Sun May 17 10:59:45 2020

@author: oislen
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def extract_model_cols(dataset):
    
    """
    
    Extract Model Columns
    
    """
    
    print('extracting out dataset columns ...')
    
    # extract the dataset columns
    data_cols = dataset.columns.tolist()
    
    # seperate out the index, target and predictor columns
    index_cols = ['primary_key', 
                  'ID', 
                  'data_split',
                  'meta_level',
                  'holdout_subset_ind',
                  'no_sales_hist_ind', 
                  'no_holdout_sales_hist_ind'
                  ]
    
    tar_cols = ['item_cnt_day']
    
    # the columns below contain information which would forward bias the results (data leakage from target)
    excl_cols = ['shop_id_total_item_cnt_day', 
                 'item_id_total_item_cnt_day',
                 'item_category_id_total_item_cnt_day',
                 'shop_id_item_category_id_total_item_cnt_day',
                 'city_enc_total_item_cnt_day',
                 'item_id_city_enc_total_item_cnt_day'
                 ]
    
    # extract the predictor columns which are not an element of index, target or exlusion columns
    pred_cols = [col for col in data_cols if col not in index_cols + tar_cols + excl_cols]
    
    # create a dictionary of the output columns
    model_cols_dict = {'index_cols':index_cols,
                       'tar_cols':tar_cols,
                       'excl_cols':excl_cols,
                       'pred_cols':pred_cols
                       }
    
    return model_cols_dict
    

def gen_cv_splits(dataset,
                  train_cv_split_dict
                  ):
    
    """
    """
    data = dataset.copy(True)
    
    cv_list = []
    
    for splits_limits in train_cv_split_dict:
        
        print(splits_limits)
        print('taking subset of data ...')
        
        sub_cols = ['date_block_num', 'data_split', 'primary_key']
        data_sub = data[sub_cols]
        
        print('creating data split sub column ...')
        
        # extract out the train, valid and test subset limits
        trn_sl = splits_limits['train_sub']
        vld_sl = splits_limits['valid_sub']
        
        print('extracting data splits ...')
    
        # define data split filters
        filt_train = data_sub['date_block_num'] <= trn_sl
        filt_valid = data_sub['date_block_num'] == vld_sl
            
        # extract out the data splits
        train_data = data_sub[filt_train].index.values.astype(int)
        valid_data = data_sub[filt_valid].index.values.astype(int)
    
        print('creating output dictionary ...')
    
        # create a dictionary of data splits
        data_splits_dict = {'train_data_idx':train_data,
                            'valid_data_idx':valid_data
                            }

        cv_list.append((data_splits_dict['train_data_idx'], data_splits_dict['valid_data_idx']))
    

    return cv_list



def gen_cv_sum(clf, cv_sum_fpath):
    
    """
    
    Generate Cross Validation Summary
    
    
    
    """
    
    cv_results = pd.DataFrame({'params':clf.cv_results_['params'],
                               'mean_test_score':clf.cv_results_['mean_test_score'] * -1,
                               'std_test_score':clf.cv_results_['std_test_score'],
                               'rank_test_score':clf.cv_results_['rank_test_score']
                               })
    
    cv_results = cv_results.sort_values(by = ['rank_test_score'])
    
    cv_results.to_csv(cv_sum_fpath, index = False)
    
    return cv_results
    


def extract_data_splits(dataset,
                        index_cols,
                        req_cols,
                        tar_cols,
                        pred_cols,
                        test_split_dict
                        ):
    
    """
    """

    data = dataset.copy(True)

    print('generating data splits ...')
    
    print('creating data split sub column ...')
    
    # extract out the train, valid and test subset limits
    trn_sl = test_split_dict['train_sub']
    vld_sl = test_split_dict['valid_sub']
    tst_sl = test_split_dict['test_sub']
    
    # error handling
    cons_holdout = (trn_sl != 34) and (vld_sl != 34) and (tst_sl != 34)
    if not cons_holdout:
        mess = 'Input Error: the specified holdout subset limit is duplicated in the train, validation or test subset limits'
        raise ValueError(mess)
    cons_sl = (trn_sl < vld_sl) and (vld_sl < tst_sl) and (tst_sl < 34)
    if not cons_sl:
        mess = 'Input Error: the specified train subset limits {}, validation subset limits {} and test subset limits {} are not time orientated.'.format(trn_sl, vld_sl, tst_sl)
        raise ValueError(mess)
        
    print('extracting data splits ...')
    
    # define data split filters
    filt_train = data['date_block_num'] <= trn_sl
    filt_valid = (data['date_block_num'] > trn_sl) & (data['date_block_num'] <= vld_sl)
    filt_test = (data['date_block_num'] > vld_sl) & (data['date_block_num'] <= tst_sl)
    filt_holdout = data['date_block_num'] == 34
    filt_meta_lvl_II = data['meta_level'].isin(['level_2', 'holdout'])
        
    # extract out the data splits
    train_data = data[filt_train]
    valid_data = data[filt_valid]
    test_data = data[filt_test]
    holdout_data = data[filt_holdout]
    meta_lvl_II = data[filt_meta_lvl_II]

    # split datasets into train, valid, test and holdout
    X_train = train_data[index_cols + pred_cols]
    y_train = train_data[index_cols + req_cols + tar_cols]
    X_valid = valid_data[index_cols + pred_cols]
    y_valid = valid_data[index_cols + req_cols + tar_cols]
    X_test = test_data[index_cols + pred_cols]
    y_test = test_data[index_cols + req_cols + tar_cols]
    X_holdout = holdout_data[index_cols + pred_cols]
    y_holdout = holdout_data[index_cols + req_cols + tar_cols]
    X_meta_lvl_II = meta_lvl_II[index_cols + pred_cols]
    y_meta_lvl_II = meta_lvl_II[index_cols + req_cols + tar_cols]
    
    print('creating output dictionary ...')
    
    # create a dictionary of data splits
    data_splits_dict = {'train_data':train_data,
                        'valid_data':valid_data,
                        'test_data':test_data,
                        'holdout_data':holdout_data,
                        'meta_lvl_II':meta_lvl_II,
                        'X_train':X_train,
                        'y_train':y_train,
                        'X_valid':X_valid,
                        'y_valid':y_valid,
                        'X_test':X_test,
                        'y_test':y_test,
                        'X_holdout':X_holdout,
                        'y_holdout':y_holdout,
                        'X_meta_lvl_II':X_meta_lvl_II,
                        'y_meta_lvl_II':y_meta_lvl_II
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
                     feat_imp, 
                     n = 20
                     ):
    
    """
    
    Extract Feature Importance Documentation
    
    """
    
    # input error handling
    # feat_imp
    valid_feat_imp = ['randforest', 'gradboost' ]
    if feat_imp not in valid_feat_imp:
        mess = 'Input Error: specified feat_imp parameter {} must be one of {}'.format(feat_imp, valid_feat_imp)
        raise ValueError(mess)
    # n features
    valid_n = type(n) == int and n > 0
    if  not (valid_n):
        mess = 'Input Error: specified n paramter {} must be a postive integer'.format(n)
        raise ValueError(mess)
    
    # set predefined index, target and required predictor columns
    index_cols = ['primary_key', 'ID', 'data_split', 'meta_level', 'holdout_subset_ind', 'no_sales_hist_ind']
    tar_cols = ['item_cnt_day']
    req_cols = ['year', 'month', 'date_block_num', 'item_id', 'shop_id']
    
    # load in feature importance file
    if feat_imp == 'randforest':
        feat_imp_df= pd.read_csv(cons.randforest_feat_imp)
    elif feat_imp == 'gradboost':
        feat_imp_df = pd.read_csv(cons.gradboost_feat_imp)
    
    # extract the required n features
    feat_imp_cols = feat_imp_df['attr'].head(n).tolist()
    
    # return unique set of predictors
    pred_cols = list(set(feat_imp_cols + req_cols))
    
    # create a dictionary of the output columns
    model_cols_dict = {'index_cols':index_cols,
                       'req_cols':req_cols,
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

def calc_rmse(dataset, tar, pred, out_fpath = None):
    """
    """
    data = dataset.copy(True)
    # calculate RMSE
    rmse = np.sqrt(((data[tar] - data[pred]) ** 2).sum() / data.shape[0])
    rmse_dict = {'RMSE':[rmse]}
    rmse_df = pd.DataFrame(rmse_dict, index = [0])
    if out_fpath != None:
        rmse_df.to_csv(out_fpath, index = False)
    return rmse_df

def plot_preds_vs_true(dataset, tar, pred, model_name, out_fpath = None):
     """
     """
     data = dataset.copy(True)
     sns.scatterplot(x = tar, y = pred, data = data)
     plt.title(model_name)
     if out_fpath != None:
         plt.savefig(out_fpath)
     plt.show() 
     return
 
def plot_preds_hist(dataset, pred, model_name, bins = 100, kde = False, out_fpath = None):
    """
    """
    data = dataset.copy(True)
    # create a hist of pred distribution
    sns.distplot(a = data[pred], bins = bins, kde = kde)
    plt.title(model_name)
    if out_fpath != None:
        plt.savefig(out_fpath)
    plt.show() 
    return