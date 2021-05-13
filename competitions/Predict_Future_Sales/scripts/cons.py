# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 17:15:06 2020

@author: oislen
"""

# import relevant libraries
import os
import sys
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
import numpy as np


# set competition name
comp_name = 'competitive-data-science-predict-future-sales'
download_data = True
unzip_data = True
del_zip = True

# set project directories
git_dir = 'C:/Users/User/Documents/GitHub'
#git_dir = '/run'
root_dir = os.path.join(git_dir, 'Kaggle')
comp_dir = os.path.join(root_dir, 'competitions/Predict_Future_Sales')
data_dir = os.path.join(comp_dir, 'data')
scripts_dir = os.path.join(comp_dir, 'scripts')
reports_dir = os.path.join(comp_dir, 'report')
models_dir = os.path.join(comp_dir, 'models')
utilities_dir = os.path.join(root_dir, 'utilities')
utilities_comp = os.path.join(utilities_dir, 'comp')
utilities_graph = os.path.join(utilities_dir, 'graph')
utilities_model = os.path.join(utilities_dir, 'model')
utilties_preproc = os.path.join(utilities_dir, 'preproc')

# set custom function location
va_dir = os.path.join(git_dir, 'value_analysis')

# set data directories
raw_data_dir = os.path.join(data_dir, 'raw')
clean_data_dir = os.path.join(data_dir, 'clean')
base_data_dir = os.path.join(data_dir, 'base')
model_data_dir = os.path.join(data_dir, 'model')
pred_data_dir = os.path.join(data_dir, 'pred')
ref_data_dir = os.path.join(data_dir, 'ref')

# set report sub directories
feat_imp_dir = os.path.join(reports_dir, 'feat_imp')
cv_results_dir = os.path.join(reports_dir, 'cv_results')
valid_plots_dir = os.path.join(reports_dir, 'valid_plots')
valid_metrics_dir = os.path.join(reports_dir, 'valid_metrics')

# set the sub validation plot / stats directories
valid_preds_hist_dir = os.path.join(valid_plots_dir, 'preds_hist')
valid_preds_vs_true_dir = os.path.join(valid_plots_dir, 'preds_vs_true')

# set raw data file paths
item_categories_fpath = os.path.join(raw_data_dir, 'item_categories.csv')
items_fpath = os.path.join(raw_data_dir, 'items.csv')
sales_train_fpath = os.path.join(raw_data_dir, 'sales_train.csv')
sample_submission_fpath = os.path.join(raw_data_dir, 'sample_submission.csv')
shops_fpath = os.path.join(raw_data_dir, 'shops.csv')
test_fpath = os.path.join(raw_data_dir, 'test.csv')

# set clean file paths
item_categories_clean_fpath = os.path.join(clean_data_dir, 'item_categories_clean.feather')
items_clean_fpath = os.path.join(clean_data_dir, 'items_clean.feather')
sales_train_clean_fpath = os.path.join(clean_data_dir, 'sales_train_clean.feather')
sample_submission_clean_fpath = os.path.join(clean_data_dir, 'sample_submission_clean.feather')
shops_clean_fpath = os.path.join(clean_data_dir, 'shops_clean.feather')
test_clean_fpath = os.path.join(clean_data_dir, 'test_clean.feather')

# set base file path
base_raw_data_fpath = os.path.join(base_data_dir, 'base_raw_data.feather')
base_raw_test_fpath = os.path.join(base_data_dir, 'base_raw_test.feather')
base_agg_data_fpath = os.path.join(base_data_dir, 'base_agg_data.feather')
base_agg_comp_fpath = os.path.join(base_data_dir, 'base_comp_data.feather')
base_agg_totl_fpath = os.path.join(base_data_dir, 'base_totl_data.feather')
base_agg_shft_fpath = os.path.join(base_data_dir, 'base_shft_data.feather')
base_agg_supp_fpath = os.path.join(base_data_dir, 'base_supp_data.feather')

# set model data file path
model_data_fpath = os.path.join(model_data_dir, 'model_data.feather')

# set feature importance paths
randforest_feat_imp = os.path.join(feat_imp_dir, 'randforest_feat_imp.csv')
gradboost_feat_imp = os.path.join(feat_imp_dir, 'gradboost_feat_imp.csv')

# set directory to pickled test shop item id combinations
holdout_shop_item_id_comb = os.path.join(ref_data_dir, 'holdout_shop_item_id_comb.pickle')

# script directories
prg_run_ensemble_dir = os.path.join(scripts_dir, '02_prg_run_ensemble')
meta_level_I_dir = os.path.join(prg_run_ensemble_dir, 'meta_level_I')

# append utilities directory to path
for p in [utilities_comp, utilities_graph, utilities_model, utilties_preproc, prg_run_ensemble_dir, meta_level_I_dir]:
    sys.path.append(p)

################
#-- Ensemble --#
################

# set model pk output file path
model_name = '{model_type}_dept{max_dept}'
model_pk_fpath = '{models_dir}/{model_name}_model.pkl'
cv_sum_fpath = '{cv_results_dir}/{model_name}_cv_summary.csv'

# assign decision tree regressor
model_dict={'dtree':DecisionTreeRegressor(),
            'gradboost':GradientBoostingRegressor(),
            'knn':KNeighborsRegressor(),
            'randforest':RandomForestRegressor()
            }

# set max number of features
n = 30

# set model parameters
params_dict = {'dtree':{'criterion':['mse', 'friedman_mse'],
                     'splitter':['best'],
                     'min_samples_split':[2, 4, 8],
                     'min_samples_leaf':[2, 4, 8],
                     'max_features':[np.int8(np.floor(n / i)) for i in [1, 2, 3, 4]],
                     },
               'gradboost':{'criterion':['friedman_mse', 'mse'],
                            'max_features':[np.int8(np.floor(n / i)) for i in [1, 2, 3, 4, 5]],
                            'n_estimators':[25, 30, 35, 40]
                            },
               'knn':{'n_neighbors':[3, 4, 5, 6, 7],
                      'weights':['uniform', 'distance'],
                      'algorithm':['auto'],
                      'p':[1, 2, 3, 4]
                      },
               'randforest':{'criterion':['mse'],
                             'n_estimators':[25, 30, 35, 40],
                             'max_features':[np.int8(np.floor(n / i)) for i in [1, 2, 3, 4, 5]]
                             }
              }


# set the train, valid and test sub limits
#train_cv_split_dict = [{'train_sub':idx, 'valid_sub':idx + 1} for idx in np.arange(start = 12, stop = 29, step = 5)]
train_cv_split_dict = [{'train_sub':28, 'valid_sub':29}]

# set the train, valid and test sub limits
test_split_dict = {'train_sub':29, 'valid_sub':32, 'test_sub':33}

# set predictions
mod_preds = '{pred_data_dir}/{model_name}_{date}'
