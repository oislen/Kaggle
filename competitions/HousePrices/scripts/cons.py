# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 10:42:07 2021

@author: oislen
"""

# load libraries
import os
import sys
from sklearn import ensemble
from sklearn.linear_model import ElasticNet, Lasso
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
import xgboost as xgb
import lightgbm as lgb

# set programme constants
comp_name = 'house-prices-advanced-regression-techniques'
download_data = True
unzip_data = True
del_zip = True

# set .csv constants
sep = ','
encoding = 'utf-8'
header = True
index = False

###########################
#-- File Path Constants --#
###########################

# set directories
git_dir = 'C:\\Users\\User\\Documents\\GitHub'
root_dir = os.path.join(git_dir, 'Kaggle')
va_dir = os.path.join(git_dir, 'value_analysis')
utilities_dir = os.path.join(root_dir, 'utilities')
houseprices_comp_dir = os.path.join(root_dir, 'competitions\\HousePrices')
scripts_dir = os.path.join(houseprices_comp_dir, 'scripts')
data_dir = os.path.join(houseprices_comp_dir, 'data')
preds_dir = os.path.join(data_dir, 'preds')
report_dir = os.path.join(houseprices_comp_dir, 'report')
metrics_dir = os.path.join(report_dir, 'model_metrics')
utilities_comp = os.path.join(utilities_dir, 'comp')
utilities_graph = os.path.join(utilities_dir, 'graph')
utilities_model = os.path.join(utilities_dir, 'model')
utilties_preproc = os.path.join(utilities_dir, 'preproc')

# create file names
train_data_fname = 'train.csv'
test_data_fname = 'test.csv'
base_data_fname = 'base.csv'
clean_data_fname = 'clean.csv'
engin_data_fname = 'engin.csv'
glm_feat_imp_fname = 'GLM_feat_imp.csv'
preds_data_fname = 'preds.csv'

# create file paths
train_data_fpath = os.path.join(data_dir, train_data_fname)
test_data_fpath = os.path.join(data_dir, test_data_fname)
base_data_fpath = os.path.join(data_dir, base_data_fname)
clean_data_fpath = os.path.join(data_dir, clean_data_fname)
engin_data_fpath = os.path.join(data_dir, engin_data_fname)
preds_data_fpath = os.path.join(data_dir, preds_data_fname)
glm_feat_imp_fpath = os.path.join(data_dir, glm_feat_imp_fname)

# append utilities directory to path
for p in [utilities_comp, utilities_graph, utilities_model, utilties_preproc]:
    sys.path.append(p)

######################
#-- Plot Constants --#
######################

plot_size_width = 12
plot_size_height = 8
plot_title_size = 25
plot_axis_text_size = 20
plot_label_size = 'x-large'

##########################
#-- Cleaning Constants --#
##########################
    
# define reference columns
Id_col = 'Id'
y_col = 'logSalePrice'
ref_cols = ['Id', 'Dataset', 'logSalePrice']
tar_cols = ['Id', 'Dataset', 'SalePrice', 'logSalePrice']

# columns to fill in zeros for
zero_imp = ['MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath', 'GarageCars', 'GarageArea']

# columns to fill in medians for
median_imp = ['LotFrontage']

# columns to impute mode for
mode_imp = ['MSZoning', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'BsmtFinType1', 'BsmtFinType2', 'Electrical', 'Functional', 'GarageType', 'SaleType', 'GarageYrBlt', 'KitchenQual',  'BsmtExposure', 'BsmtQual', 'GarageQual', 'GarageCond',  'Utilities', 'GarageFinish', 'BsmtCond']

# columns to impute new category 'none' for
none_imp = ['MiscFeature', 'FireplaceQu', 'PoolQC', 'Alley', 'Fence']

# create a list of quality columns with the same levels
QC_Cols = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC', 'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond', 'PoolQC']

# create a list of corresponding ordinal names for thee columns
QC_Ord_Cols = ['{}_Ord'.format(col) for col in QC_Cols]

# create the mapping dictionary
qc_cols_map_dict = {'Ex':5, 'Gd':4, 'TA':3, 'Fa':2, 'Po':1, 'None':0}

# create the mapping dictionary
lotshape_map_dict = {'Reg':4, 'IR1':3, 'IR2':2, 'IR3':1}

# create the mapping dictionary
landslope_map_dict = {'Gtl':1, 'Mod':2, 'Sev':3}

# create the mapping dictionary
basement_expo_map_dict = {'Gd':4, 'Av':3, 'Mn':2, 'No':1}

# create the mapping dictionary
garagefinish_map_dict = {'Fin':3, 'RFn':2, 'Unf':1}

# create the mapping dictionary
paved_drive_map_dict = {'Y':3, 'P':2, 'N':1}

# create the mapping dictionary
alley_map_dict = {'None':0, 'Grvl':1, 'Pave':2}

# create the mapping dictionary
utilities_map_dict = {'ELO':1, 'NoSeWa':2, 'NoSewr':3, 'AllPub':4}

# create dummy columns list
dummy_cols = ['MSZoning', 'Street', 'LandContour', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'Foundation', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 'CentralAir', 'Electrical', 'Functional', 'GarageType', 'Fence', 'MiscFeature', 'SaleType', 'SaleCondition']

# create outlier dictionary for filtering out anomalies
outlier_dict = {'1stFlrSF':4000, 'BsmtFinSF1':4000, 'GrLivArea':5000, 'LotFrontage':300, 'MiscVal':8000, 'OpenPorchSF':500, 'TotalBsmtSF':6000, 'SalePrice':300000}

#######################
#-- Model Constants --#
#######################

# set the random state
random_state = 1234

# set target type for perf_metrics
target_type = 'reg'

# define a random forest regressor model
rfr_mod = ensemble.RandomForestRegressor(random_state = random_state)

# create LASSO model
lasso = Lasso(alpha = 0.0005, 
              random_state = 1
              )

# create elastic net model
ENet =  ElasticNet(alpha = 0.0005, 
                   l1_ratio = 0.9, 
                   random_state = 3
                   )

# create kernel ridge regression model
KRR = KernelRidge(alpha = 0.6, 
                  kernel = 'polynomial', 
                  degree = 2, 
                  coef0 = 2.5
                  )

# create gradient boosted model
GBoost = GradientBoostingRegressor(n_estimators = 3000, 
                                   learning_rate = 0.05,
                                   max_depth = 4, 
                                   max_features = 'sqrt',
                                   min_samples_leaf = 15, 
                                   min_samples_split = 10, 
                                   loss = 'huber', 
                                   random_state =5
                                   )

# create XGBoost model
model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.05, max_depth=3, 
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1)

# create LGBoost model
model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)

# create a dictionary of models
models_dict = {'lasso':lasso, 
               'ENet':ENet, 
               'KRR':KRR, 
               'GBoost':GBoost, 
               'model_xgb':model_xgb, 
               'model_lgb':model_lgb
               }
