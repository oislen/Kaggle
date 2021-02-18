# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 09:58:04 2019

@author: oislen
"""

# load in the relevant libraries
import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNet, Lasso
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
import xgboost as xgb
import lightgbm as lgb

# load cusotm functions
import sys
va_dir = 'C:/Users/User/Documents/Data_Analytics/Python/value_analysis'
sys.path.append(va_dir)
import value_analysis as va

# load in data
kaggle_dir = 'C:/Users/User/Documents/GitHub/Kaggle/HousePrices/'
input_dir = 'data/'
metrics_dir = 'report/model_metrics/'

clean_name = 'clean.csv'
clean = pd.read_csv(kaggle_dir + input_dir + clean_name, 
                    sep = '|'
                    )

# create the output directory
output_dir = kaggle_dir + input_dir + 'python_predictions/'

"""
########################
#-- Standardise data --#
########################
"""

print('Standardising data ...')

# define reference columns
ref_cols = ['Id', 'Dataset', 'logSalePrice']

# create the standardisation dataset
stand = pd.DataFrame()
stand[ref_cols] = clean[ref_cols]

# create a list of variables to standardise
stand_cols = clean.columns.drop(ref_cols).tolist()
#stand_cols = ['BsmtFinSF2', 'LowQualFinSF', 'BsmtHalfBath']

# standardise the dataset using the robust scalar method
stand[stand_cols] = va.standardise_variables(dataset = clean,
                                             attr = stand_cols,
                                             stand_type = 'robust'
                                             )

"""
#########################
#-- Split the Dataset --#
#########################
"""

print('Splitting data ...')

# extract out the test set id column
test_ID = clean.loc[(clean['Dataset'] == 'test'), 'Id']

# split the datasets
test = clean[(clean['Dataset'] == 'test')]
train = clean[(clean['Dataset'] == 'train')]

# randomly split the dataset
X_valid, X_train, y_valid, y_train = va.train_test_split_sample(dataset = train,
                                                                y = ['logSalePrice'],
                                                                X = train.columns.drop(['Id', 'logSalePrice', 'Dataset']),
                                                                train_size = 0.8,
                                                                test_size = 0.2
                                                                )

# extract out the relevant columns
y_train_sub = y_train.logSalePrice.values
y_valid_sub = y_valid.logSalePrice.values
X_test = test.drop(columns = ['Id', 'logSalePrice', 'Dataset'])

"""
####################
#-- Create Model --#
####################
"""

# create LASSO model
lasso = Lasso(alpha =0.0005, random_state=1)

# create elastic net model
ENet =  ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3)

# create kernel ridge regression model
KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)

# create gradient boosted model
GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10, 
                                   loss='huber', random_state =5)

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

"""
###################
#-- Tune Models --#
###################
"""
#-- Lasso Model --#

# create parameters dictionary
params_dict = {'alpha':[0.0001, 0.0002, 0.0005, 0.0008,
                        0.001, 0.002, 0.005, 0.008,
                        0.01, 0.02, 0.05, 0.08,
                        0.1, 0.2, 0.5, 0.8
                        ]
               }

# tune the parameters
va.tune_hyperparameters(model = lasso, 
                        params = params_dict,
                        X_train = X_train,
                        y_train = y_train,
                        scoring = 'neg_mean_squared_error'
                        )

"""
################################
#-- Individual Model Metrics --#
################################
"""

# create a list of models
models = [lasso, ENet, KRR, GBoost, model_xgb, model_lgb]
models_pref = ['lasso', 'ENet', 'KRR', 'GBoost', 'model_xgb', 'model_lgb']

# for each model
for idx, model in enumerate(models):
    
    # extract the model prefix
    pref = models_pref[idx]
    
    print('Working on ' + pref)
    
    # generate cross fold validation metrics
    va.metrics(model = model, 
               endog = y_train_sub,
               exog = X_train,
               target_type = 'reg',
               n_folds = 5,
               digits = 3,
               output_dir = kaggle_dir + metrics_dir,
               output_fname = pref + '_model_metrics.csv'
               )

"""
#####################
#-- Stacked Model --#
#####################
"""

#-- Stack Model

# create the model (via stacked class)
StackMod = va.StackModels(base_models = (ENet, GBoost, KRR),
                          meta_model = lasso
                          )

# fit the model
StackMod.fit(X_train.values, y_train_sub)

# predict for the validation values
sm_y_valid_pred = StackMod.predict(X_valid.values)

# generate metrics for the predictions
va.metrics(y_obs = y_valid['logSalePrice'].values,
           y_pred = sm_y_valid_pred,
           target_type = 'reg',
           digits = 4
           )

#-- XG Boost Model --#

# fit the xgboost mode
model_xgb.fit(X_train.values, y_train_sub)

# predict for the validation values
xgb_y_valid_pred = model_xgb.predict(X_valid.values)

# generate metrics for the predictions
va.metrics(y_obs = y_valid['logSalePrice'].values,
           y_pred = xgb_y_valid_pred,
           target_type = 'reg',
           digits = 4
           )

#-- LG Boost Model --#

# fit the lgboost model
model_lgb.fit(X_train.values, y_train_sub)

# predict of the validation values
lgb_y_valid_pred = model_lgb.predict(X_valid.values)

# generate metrics for the predictions
va.metrics(y_obs = y_valid['logSalePrice'].values,
           y_pred = xgb_y_valid_pred,
           target_type = 'reg',
           digits = 4
           )

###################################
#-- Create Final Ensemble Model --#
###################################

drop_cols = ['Id', 'Dataset', 'logSalePrice']
X_train_full = train.drop(columns = drop_cols).values
y_train_full = train['logSalePrice'].values
X_test = test.drop(columns = drop_cols).values

# create the stacked model predictions
StackMod.fit(X_train_full, y_train_full)
sm_pred = np.exp(StackMod.predict(X_test))

# create the xgboost model predictions
model_xgb.fit(X_train_full, y_train_full)
xgb_pred = np.exp(model_xgb.predict(X_test))

# create the lgboost model predictions
model_lgb.fit(X_train_full, y_train_full)
lgb_pred = np.exp(model_lgb.predict(X_test))

# create a weighted ensemble from the three models
ensemble = sm_pred * 0.70 + xgb_pred * 0.15 + lgb_pred * 0.15

sub = pd.DataFrame()
sub['Id'] = test_ID.astype(int)
sub['SalePrice'] = ensemble

# output the dataset
sub.to_csv(output_dir + 'preds1.csv',
           sep = ',',
           encoding = 'utf-8',
           header = True,
           index = False
           )