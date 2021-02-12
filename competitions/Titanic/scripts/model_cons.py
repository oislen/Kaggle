# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 16:12:58 2021

@author: oislen
"""

# import relevant libraries
from sklearn import ensemble, tree, svm

# create empty dicitonaries to hold the age and survival models
age_dict = {}  
sur_dict = {}

# set random state for models
random_state = 123

#-- Age --#

# create the age models
age_gbr_mod = ensemble.GradientBoostingRegressor(random_state = random_state) 
age_rfr_mod = ensemble.RandomForestRegressor(random_state = random_state)

# create age parameters
age_gbr_params = {'loss':['lad'], 'learning_rate':[1.0, 0.9, 0.8], 'n_estimators':[50, 100], 'max_depth':[1, 3, 5], 'max_features':['sqrt', 'log2']}
age_rfr_params = {'criterion':['mse', 'mae'], 'n_estimators':[100], 'min_samples_split':[2, 3], 'max_features':['auto', 'sqrt', 'log2'], 'n_jobs':[-1]}

# create model and param dictionaries
age_dict['gbr'] = {'model':age_gbr_mod, 'params':age_gbr_params}
age_dict['rfr'] = {'model':age_rfr_mod, 'params':age_rfr_params}

#-- Survival --#

# create survival models
sur_gbc_mod = ensemble.GradientBoostingClassifier(random_state = random_state)
sur_rfc_mod = ensemble.RandomForestClassifier(random_state = random_state)
sur_dtc_mod = tree.DecisionTreeClassifier(random_state = random_state)
sur_abc_mod = ensemble.AdaBoostClassifier(base_estimator = sur_dtc_mod, random_state = random_state)
sur_etc_mod = ensemble.ExtraTreesClassifier(random_state = random_state)
sur_svc_mod = svm.SVC(random_state = random_state)
#sur_knc_mod = neighbors.KNeighborsClassifier()
#sur_lrc_mod = linear_model.LogisticRegression(random_state = random_state)
#sur_nbc_mod = naive_bayes.GaussianNB()

# create survival parameters
sur_gbc_params = {'loss' : ["deviance"], 'n_estimators' : [100, 200, 300], 'learning_rate': [0.1, 0.05, 0.01], 'max_depth': [4, 8], 'min_samples_leaf': [100, 150], 'max_features': [0.3, 0.1]}
sur_rfc_params ={"max_depth": [None], "max_features": [1, 3, 10],  "min_samples_split": [2, 3, 10], "min_samples_leaf": [1, 3, 10], "bootstrap": [False], "n_estimators" :[100, 300], "criterion": ["gini"]}
sur_dtc_parmas = {'criterion':['gini', 'entropy'], 'splitter':['best', 'random'],'min_samples_split':[2, 3]}
sur_abc_params = {"base_estimator__criterion" : ["gini", "entropy"], "base_estimator__splitter" :   ["best", "random"], "algorithm" : ["SAMME","SAMME.R"], "n_estimators" :[1,2], "learning_rate":  [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3,1.5]}
sur_etc_params = {"max_depth": [None], "max_features": [1, 3, 10], "min_samples_split": [2, 3, 10], "min_samples_leaf": [1, 3, 10], "bootstrap": [False], "n_estimators" :[100,300], "criterion": ["gini"]}
sur_svc_params = {'kernel': ['rbf'], 'gamma': [ 0.001, 0.01, 0.1, 1], 'C': [1, 10, 50, 100,200,300, 1000]}

#sur_knc_params = {'n_neighbors':[3, 5, 7], 'weights':['uniform', 'distance'], 'algorithm':['auto'], 'p':[1, 2, 3], 'n_jobs':[-1]}
#sur_lrc_params = {'penalty':['l1', 'l2', 'elasticnet', 'none'], 'solver':['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'], 'n_jobs':[-1]}
#sur_nbc_params = {}

# create model and param dictionaries
sur_dict['gbc'] = {'model':sur_gbc_mod, 'params':sur_gbc_params}
sur_dict['rfc'] = {'model':sur_rfc_mod, 'params':sur_rfc_params}
sur_dict['dtc'] = {'model':sur_dtc_mod, 'params':sur_dtc_parmas}
sur_dict['abc'] = {'model':sur_abc_mod, 'params':sur_abc_params}
sur_dict['etc'] = {'model':sur_etc_mod, 'params':sur_etc_params}
sur_dict['svc'] = {'model':sur_svc_mod, 'params':sur_svc_params}
#sur_dict['knc'] = {'model':sur_knc_mod, 'params':sur_knc_params}
#sur_dict['lrc'] = {'model':sur_lrc_mod, 'params':sur_lrc_params}
#sur_dict['nbc'] = {'model':sur_nbc_mod, 'params':sur_nbc_params}



