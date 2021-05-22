# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 17:15:06 2020

@author: oislen
"""

# import relevant libraries
import os
import sys
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
import numpy as np

# set pandas options
pd.set_option('display.max_columns', 10)

# set competition name
comp_name = 'competitive-data-science-predict-future-sales'
download_data = True
unzip_data = True
del_zip = True

# set project directories
git_dir = 'C:\\Users\\User\\Documents\\GitHub'
#git_dir = '/run'
root_dir = os.path.join(git_dir, 'Kaggle')
comp_dir = os.path.join(root_dir, 'competitions\\Predict_Future_Sales')
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

# set model data file paths
model_data_fpath = os.path.join(model_data_dir, 'model_data.feather')
randforest_feat_imp = os.path.join(feat_imp_dir, 'randforest_feat_imp.csv')
gradboost_feat_imp = os.path.join(feat_imp_dir, 'gradboost_feat_imp.csv')
meta_feat_fpath = os.path.join(model_data_dir, 'meta_feats.feather')

# set validation file path
model_name = '{model_type}'
preds_vs_true_fpath = os.path.join(valid_preds_vs_true_dir, model_name)
preds_hist_fpath = os.path.join(valid_preds_hist_dir, model_name)
preds_metrics_fpath = os.path.join(valid_metrics_dir, model_name)
mod_preds_fpath = os.path.join(pred_data_dir, model_name)

# set the validation output paths and set the prediction output paths
preds_valid_rmse = '{preds_metrics_fpath}_valid_rmse.csv'.format(preds_metrics_fpath = preds_metrics_fpath)
preds_test_rmse = '{preds_metrics_fpath}_test_rmse.csv'.format(preds_metrics_fpath = preds_metrics_fpath)
preds_vs_true_valid = '{preds_vs_true_fpath}_preds_vs_true_valid.png'.format(preds_vs_true_fpath = preds_vs_true_fpath)
preds_vs_true_test = '{preds_vs_true_fpath}_preds_vs_true_test.png'.format(preds_vs_true_fpath = preds_vs_true_fpath)
true_hist_valid = '{preds_hist_fpath}_true_valid.png'.format(preds_hist_fpath = preds_hist_fpath)
true_hist_test = '{preds_hist_fpath}_true_test.png'.format(preds_hist_fpath = preds_hist_fpath)
preds_hist_valid = '{preds_hist_fpath}_preds_valid.png'.format(preds_hist_fpath = preds_hist_fpath)
preds_hist_test = '{preds_hist_fpath}_preds_test.png'.format(preds_hist_fpath = preds_hist_fpath)
preds_hist_holdout = '{preds_hist_fpath}_preds_holdout.png'.format(preds_hist_fpath = preds_hist_fpath)
y_valid_preds_path = '{mod_preds_fpath}_valid.feather'.format(mod_preds_fpath = mod_preds_fpath)
y_test_preds_path = '{mod_preds_fpath}_test.feather'.format(mod_preds_fpath = mod_preds_fpath)
y_holdout_preds_path = '{mod_preds_fpath}_holdout.feather'.format(mod_preds_fpath = mod_preds_fpath)
meta_lvl_II_feats_path = '{mod_preds_fpath}_meta_lvl_II_feats.feather'.format(mod_preds_fpath = mod_preds_fpath)
kaggle_preds = '{mod_preds_fpath}.csv'.format(mod_preds_fpath = mod_preds_fpath)

# create a dictionary for the validation output file paths
result_output_paths = {'preds_valid_rmse':preds_valid_rmse,
                      'preds_test_rmse':preds_test_rmse,
                      'preds_vs_true_valid':preds_vs_true_valid,
                      'preds_vs_true_test':preds_vs_true_test,
                      'true_hist_valid':true_hist_valid,
                      'true_hist_test':true_hist_test,
                      'preds_hist_valid':preds_hist_valid,
                      'preds_hist_test':preds_hist_test,
                      'preds_hist_holdout':preds_hist_holdout,
                      'y_valid_preds_path':y_valid_preds_path,
                      'y_test_preds_path':y_test_preds_path,
                      'y_holdout_preds_path':y_holdout_preds_path,
                      'meta_lvl_II_feats_path':meta_lvl_II_feats_path,
                      'kaggle_preds':kaggle_preds
                      }
    
# set model pk output file path
model_pk_fpath = os.path.join('{models_dir}', '{model_name}_model.pkl')
cv_sum_fpath = os.path.join('{cv_results_dir}', '{model_name}_cv_summary.csv')

# set output path for meta-level II predictions
meta_level_II_preds_fpath = os.path.join(pred_data_dir, 'stackmodel.csv')

# set directory to pickled test shop item id combinations
holdout_shop_item_id_comb = os.path.join(ref_data_dir, 'holdout_shop_item_id_comb.pickle')

# script directories
prg_run_meta_level_I = os.path.join(scripts_dir, '02_prg_run_meta_level_I')
prg_run_meta_level_II = os.path.join(scripts_dir, '03_prg_run_meta_level_II')
meta_level_I_reference_path = os.path.join(prg_run_meta_level_I, 'reference')

# append utilities directory to path
for p in [utilities_comp, utilities_graph, utilities_model, utilties_preproc, prg_run_meta_level_I, meta_level_I_reference_path, prg_run_meta_level_II]:
    sys.path.append(p)

######################
#-- Plot Constants --#
######################

plot_size_width = 12
plot_size_height = 8
plot_title_size = 25
plot_axis_text_size = 20
plot_label_size = 'x-large'
bins = 100
kde = False

##########################
#-- Feature Importance --#
##########################

feat_imp_max_depth = 7
feat_imp_n_estimators = 20
feat_imp_criterion = 'mse'
feat_imp_max_features = 'auto'

####################
#-- Meta-Level I --#
####################

# execution constants
n_cpu = -1
verbose = 3
refit_bool = True
max_dept = 10
lower_bound = 0
upper_bound = 20

# set ransom seed
rand_seed = 1234
np.random.seed(rand_seed)

# assign decision tree regressor
model_dict={'dtree':DecisionTreeRegressor(),
            'gradboost':GradientBoostingRegressor(),
            'knn':KNeighborsRegressor(),
            'randforest':RandomForestRegressor()
            }

# set max number of features
n = 43

# set model parameters
params_dict = {'dtree':{'criterion':['mse'],
                     'splitter':['best'],
                     'min_samples_split':[2, 4, 8],
                     'min_samples_leaf':[2, 4, 8],
                     #'max_features':[np.int8(np.float8(n / i)) for i in [1, 2, 3, 4]],
                     'random_state':[rand_seed],
                     'max_depth':[10]
                     },
               'gradboost':{'criterion':['mse'],
                            #'max_features':[np.float8(np.floor(n / i)) for i in [1, 2, 3, 4]],
                            'n_estimators':[25],
                            'random_state':[rand_seed],
                            'max_depth':[7]
                            },
               'randforest':{'criterion':['mse'],
                             'n_estimators':[25],
                             #'max_features':[np.float8(np.floor(n / i)) for i in [1, 2, 3, 4]],
                             'random_state':[rand_seed],
                             'max_depth':[9]
                             }
              }


# set the train, valid and test sub limits
#train_cv_split_dict = [{'train_sub':idx, 'valid_sub':idx + 1} for idx in np.arange(start = 12, stop = 29, step = 5)]
train_cv_split_dict = [{'train_sub':28, 'valid_sub':29}]

# set the train, valid and test sub limits
test_split_dict = {'train_sub':29, 'valid_sub':32, 'test_sub':33}

#####################
#-- Meta-Level II --#
#####################

# set base columns for reading meta level II features
meta_level_II_base_cols = ['primary_key', 'ID', 'data_split', 'meta_level', 'holdout_subset_ind',
                           'no_sales_hist_ind', 'year', 'month', 'date_block_num', 'item_id',
                           'shop_id', 'item_cnt_day']

# set pred columns when reading meta level II
meta_level_II_resp_col = ['item_cnt_day']
meta_level_II_tar_cols = ['primary_key', 'y_meta_lvl_I_pred']
