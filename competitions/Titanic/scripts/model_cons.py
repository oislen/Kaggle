# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 16:12:58 2021

@author: oislen
"""

from sklearn import ensemble

# create empty dicitonaries to hold the age and survival models
age_dict = {}  
sur_dict = {}

# set random state for models
random_state = 123

# create the models
age_gbm_mod = ensemble.GradientBoostingRegressor(random_state = random_state) 
sur_gbm_mod = ensemble.GradientBoostingClassifier(random_state = random_state)
age_rf_mod = ensemble.RandomForestRegressor(random_state = random_state)

# create parameters
age_gbm_params = {'loss':['lad'], 'learning_rate':[1.0, 0.9, 0.8], 'n_estimators':[50, 100], 'max_depth':[1, 3, 5], 'max_features':['sqrt', 'log2']}
sur_gbm_params = {'loss':['deviance', 'exponential'], 'subsample':[1, 0.9, 0.8], 'learning_rate':[1.0, 0.9, 0.8], 'min_samples_split':[2, 3, 4], 'min_samples_leaf':[1, 2, 3], 'n_estimators':[25, 50], 'max_depth':[1, 3, 5]}
age_rf_params = {'criterion':['mse', 'mae'], 'n_estimators':[100], 'min_samples_split':[2, 3], 'max_features':['auto', 'sqrt', 'log2'], 'n_jobs':[-1]}

# create model and param dictionaries
age_gbm_dict = {'model':age_gbm_mod, 'params':age_gbm_params}
sur_gbm_dict = {'model':sur_gbm_mod, 'params':sur_gbm_params}
age_rf_dict = {'model':age_rf_mod, 'params':age_rf_params}

# assign to output dictionarys
age_dict['gbm'] = age_gbm_dict
sur_dict['gbm'] = sur_gbm_dict
age_dict['rf'] = age_rf_dict


