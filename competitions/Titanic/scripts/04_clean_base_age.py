# -*- coding: utf-8 -*-
"""
Created on Sun Nov  4 19:18:53 2018

@author: oislen
"""

"""
#####################
#-- Preliminaries --#
#####################
"""

print('Loading in libraries and data ...')

# load in relevant libraries
import pandas as pd
import cons

# sklearn will be used for the modelling and classification
from sklearn import ensemble

# load cusotm functions
import value_analysis as va

# load in data
base = pd.read_csv(cons.base_clean_data_fpath, sep = '|')

# check data types
base.dtypes

# split the data based on the original dataset
base_train = base[base.Dataset == 'train']
base_test = base[base.Dataset == 'test']

"""
#######################
#-- Clean Train Age --#
########################
"""

print('Cleaning Age in base trian ...')

# split the training data on whether age is missing or not
train = base_train[base_train.Age.notnull()]
test = base_train[base_train.Age.isnull()]

# randomly split the dataset
X_valid, X_train, y_valid, y_train = va.train_test_split_sample(dataset = train,
                                                                y = ['Age'],
                                                                X = train.columns.drop(['Age', 'PassengerId', 'Dataset']),
                                                                train_size = 0.8,
                                                                test_size = 0.2,
                                                                random_split = True,
                                                                sample_target = None
                                                                )

# initiate gbm
gbm = ensemble.GradientBoostingRegressor(random_state = 123)

# save the parametre features to tune as a dictionary
params = {'loss':['lad'],
          'learning_rate':[1.0, 0.9, 0.8],
          'n_estimators':[50, 100, 200],
          'max_depth':[1, 3, 5],
          'max_features':[None, 'sqrt', 'log2'],
          'presort':[True]
          }

# tune gbm model
mod_tuning = va.tune_hyperparameters(model = gbm, 
                                     params = params, 
                                     X_train = X_train, 
                                     y_train = y_train,
                                     scoring = 'neg_mean_squared_error'
                                     )

# extract the best parameters
best_params = mod_tuning.loc[0, 'params']

# initiate the best model
gbm = ensemble.GradientBoostingRegressor(learning_rate = best_params['learning_rate'],
                                         loss = best_params['loss'],
                                         max_depth = best_params['max_depth'],
                                         max_features = best_params['max_features'],
                                         n_estimators = best_params['n_estimators'],
                                         presort = best_params['presort'],
                                         random_state = 123
                                         )

# fit the best model
gbm.fit(X_train, 
        y_train.values.ravel()
        )

# classify the validation set
y_valid['Age_pred'] = gbm.predict(X_valid)

# genrate the regression metrics
va.perf_metrics(y_obs = y_valid['Age'], 
                y_pred = y_valid['Age_pred'], 
                target_type = 'reg'
                )

# create prediction, observation and residual plots
va.Vis.preds_obs_resids(obs = 'Age',
                        preds = 'Age_pred',
                        dataset = y_valid
                        )

# predict for the test set
test['Age'] = gbm.predict(test[X_valid.columns])

# re-concatenate the training and test to update the base train set
base_train = pd.concat(objs = [train, test],
                       axis = 0
                       )

"""
######################
#-- Clean Test Age --#
#######################
"""

print('Cleaning Age in base test ...')

# randomly split the dataset
X_valid, X_train, y_valid, y_train = va.train_test_split_sample(dataset = base_train,
                                                                y = ['Age'],
                                                                X = train.columns.drop(['Age', 'Survived', 'PassengerId', 'Dataset']),
                                                                train_size = 0.8,
                                                                test_size = 0.2,
                                                                random_split = True,
                                                                sample_target = None
                                                                )

# initiate gbm
gbm = ensemble.GradientBoostingRegressor(random_state = 123)

# save the parametre features to tune as a dictionary
params = {'loss':['lad'],
          'learning_rate':[1.0, 0.9, 0.8],
          'n_estimators':[50, 100, 200],
          'max_depth':[1, 3, 5],
          'max_features':[None, 'sqrt', 'log2'],
          'presort':[True]
          }

# tune gbm model
mod_tuning = va.tune_hyperparameters(model = gbm, 
                                     params = params, 
                                     X_train = X_train, 
                                     y_train = y_train,
                                     scoring = 'neg_mean_squared_error'
                                     )

# extract the best parameters
best_params = mod_tuning.loc[0, 'params']

# initiate the best model
gbm = ensemble.GradientBoostingRegressor(learning_rate = best_params['learning_rate'],
                                         loss = best_params['loss'],
                                         max_depth = best_params['max_depth'],
                                         max_features = best_params['max_features'],
                                         n_estimators = best_params['n_estimators'],
                                         presort = best_params['presort'],
                                         random_state = 123
                                         )

# fit the best model
gbm.fit(X_train, 
        y_train.values.ravel()
        )

# classify the validation set
y_valid['Age_pred'] = gbm.predict(X_valid)

# genrate the regression metrics
va.perf_metrics(y_obs = y_valid['Age'], 
                y_pred = y_valid['Age_pred'], 
                target_type = 'reg'
                )

# create prediction, observation and residual plots
va.Vis.preds_obs_resids(obs = 'Age',
                        preds = 'Age_pred',
                        dataset = y_valid
                        )

# predict for the base_test set
base_test['Age'] = gbm.predict(base_test[X_valid.columns])

# re-concatenate the base training and base test to update base data
base = pd.concat(objs = [base_train, base_test],
                 axis = 0
                 )

"""
##############
#-- Output --#
##############
"""

print('Outputting data ...')

# output the dataset
base.to_csv(cons.base_clean_2_data_fpath,
            sep = '|',
            encoding = 'utf-8',
            header = True,
            index = False
            )