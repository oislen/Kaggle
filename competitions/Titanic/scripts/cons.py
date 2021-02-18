# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 17:20:45 2021

@author: oislen
"""

# load libraries
import os
import sys
import pandas as pd
from sklearn import ensemble, tree, svm

# set pandas behaviour
pd.set_option('display.max_columns', 20)

# set .csv constants
sep = ','
encoding = 'utf-8'
header = True
index = False

##########################
#-- Filepath Constants --#
##########################

# set programme constants
comp_name = 'titanic'
download_data = True
unzip_data = True
del_zip = True

# set directories
git_dir = 'C:\\Users\\User\\Documents\\GitHub'
root_dir = os.path.join(git_dir, 'Kaggle')
utilities_dir = os.path.join(root_dir, 'utilities')
titanic_comp_dir = os.path.join(root_dir, 'competitions\\Titanic')
scripts_dir = os.path.join(titanic_comp_dir, 'scripts')
data_dir = os.path.join(titanic_comp_dir, 'data')
report_dir = os.path.join(titanic_comp_dir, 'report')
preds_dir = os.path.join(data_dir, 'preds')
models_dir = os.path.join(data_dir, 'models')
univar_stats_dir = os.path.join(report_dir, 'univariate_analysis\\descriptive_stats')
bivar_assoc_dir = os.path.join(report_dir, 'bivariate_analysis\\association_tests')
bivar_corr_dir = os.path.join(report_dir, 'bivariate_analysis\\correlation_tests')
bivar_gains_dir = os.path.join(report_dir, 'bivariate_analysis\\measure_gains')
model_results_dir = os.path.join(report_dir, 'model_results\\{}')

# define filenames
zip_data_fname = '{}.zip'.format(comp_name)
sample_sub_data_fname = 'gender_submission.csv'
test_data_fname = 'test.csv'
train_data_fname = 'train.csv'
ground_truth_fname = 'ground_truth.csv'
base_data_fname = 'base.csv'
base_clean_data_fname = 'base_clean.csv'
base_clean_2_data_fname = 'base_clean_2.csv'
base_engin_fname = 'base_engin.csv'
pred_data_fname = '{}_preds.csv'
hyperparam_fname = '{}_hyperparam_tuning.csv'
best_model_fname = '{}_best_model.pkl'
hist_train_tar_fname = '{}_hist_train_tar.png'
learning_curve_fnamt = '{}_learning_curve.png'
hist_valid_preds_fname = '{}_hist_valid_preds.png'
metrics_fname = '{}_perf_metrics.csv'
roc_fname = '{}_roc_curve.png'
feat_imp = '{}_feat_imp.png'
hist_test_preds_fname = '{}_hist_test_preds.png'
gf_fname = "geneticfunction.pickle"

# create file paths
zip_data_fpath = os.path.join(data_dir, zip_data_fname)
sample_sub_data_fpath = os.path.join(data_dir, sample_sub_data_fname)
test_data_fpath = os.path.join(data_dir, test_data_fname)
train_data_fpath = os.path.join(data_dir, train_data_fname)
ground_truth_fpath = os.path.join(data_dir, ground_truth_fname)
base_data_fpath = os.path.join(data_dir, base_data_fname)
base_clean_data_fpath = os.path.join(data_dir, base_clean_data_fname)
base_clean_2_data_fpath = os.path.join(data_dir, base_clean_2_data_fname)
base_engin_data_fpath = os.path.join(data_dir, base_engin_fname)
pred_data_fpath = os.path.join(preds_dir, pred_data_fname)
best_model_fpath = os.path.join(models_dir, best_model_fname)
gf_fpath = os.path.join(models_dir, gf_fname)

hyper_param_fpath = os.path.join(model_results_dir, hyperparam_fname)
perf_metrics_fpath = os.path.join(model_results_dir, metrics_fname)

# append utilities directory to path
for p in [utilities_dir]:
    sys.path.append(p)

# set ground truth url
url = "https://github.com/thisisjasonjafari/my-datascientise-handcode/raw/master/005-datavisualization/titanic.csv"

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

# set id columns
id_cols = ['PassengerId', 'Dataset', 'Survived']

# set base columns
drop_cols = ['Name', 'Sex', 'Ticket', 'Fare', 'Cabin', 'Embarked', 'Title', 'FamSize_Cat', 'Ticket_Prefix', 'Cabin_Letters']

# the title of a person indicates the person status
title_map = {'Mr':'Mr', 'Miss':'Ms', 'Mrs':'Mrs', 'Master':'Master', 'Dr':'Dr', 'Rev':'Rev', 'Col':'Col', 'Major':'Major', 'Ms':'Ms', 'Mlle':'Ms', 'Capt':'Capt', 'Don':'Mr', 'Jonkheer':'Jonkheer', 'Sir':'Sir', 'Lady':'Lady', 'Mme':'Mrs', 'Dona':'Mrs', 'Countess':'Countess'}

# create a map for the title values
priv_map = {'Mr':'Mr', 'Ms':'Ms', 'Mrs':'Mrs', 'Master':'Priv', 'Dr':'Priv', 'Rev':'Priv', 'Col':'Priv', 'Major':'Priv', 'Capt':'Priv', 'Jonkheer':'Priv', 'Sir':'Priv', 'Lady':'Priv', 'Countess':'Priv'}

# create ordinal mapping for title
title_ord_map = {'Mr':1, 'Ms':2, 'Mrs':3, 'Priv':4}

# create cabin map
cab_map = {'A':'A', 'B':'B', 'BB':'B', 'BBB':'B', 'BBBB':'B', 'C':'C', 'CC':'C', 'CCC':'C', 'D':'D', 'DD':'D', 'E':'E', 'EE':'E', 'F':'F', 'FG':'F', 'FE':'F', 'G':'G', 'T':'T'}

# create embarked map
embarked_map = {'S':1, 'C':2, 'Q':3}

#######################
#-- Model Constants --#
#######################

# create empty dicitonaries to hold the age and survival models
age_dict = {}  
sur_dict = {}

# settins for models
random_state = 123
train_size = 0.8
test_size = 0.2
random_split = True
refit = True
verbose = 3
cv = 10
n_jobs = -1

#-- Age --#

# define age columns
y_col_age = ['Age']
non_X_cols = ['PassengerId', 'Age', 'Survived']
X_col_age = ['Pclass', 'SibSp', 'Parch', 'FamSize', 'Fare', 'Alone', 'Mr', 'Mrs', 'Ms', 'Priv', 'Male', 'Embarked_Ord']

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

# define survival columns
y_col = ['Survived']
X_col =  ['Pclass', 'SibSp', 'Parch', 'FamSize', 'Fare_Log', 'Alone', 'Mr', 'Mrs', 'Ms', 'Priv', 'Male', 'Embarked_Ord']

# create survival models
sur_gbc_mod = ensemble.GradientBoostingClassifier(random_state = random_state)
sur_rfc_mod = ensemble.RandomForestClassifier(random_state = random_state)
sur_dtc_mod = tree.DecisionTreeClassifier(random_state = random_state)
sur_abc_mod = ensemble.AdaBoostClassifier(base_estimator = sur_dtc_mod, random_state = random_state)
sur_etc_mod = ensemble.ExtraTreesClassifier(random_state = random_state)
sur_svc_mod = svm.SVC(probability = True, random_state = random_state)

# create survival parameters
sur_gbc_params = {'loss' : ["deviance"], 'n_estimators' : [100, 200, 300], 'learning_rate': [0.1, 0.05, 0.01], 'max_depth': [4, 8], 'min_samples_leaf': [100, 150], 'max_features': [0.3, 0.1]}
sur_rfc_params ={"max_depth": [None], "max_features": [1, 3, 10],  "min_samples_split": [2, 3, 10], "min_samples_leaf": [1, 3, 10], "bootstrap": [False], "n_estimators" :[100, 300], "criterion": ["gini"]}
#sur_rfc_params ={"max_depth": [None], "max_features": [10],  "min_samples_split": [2, 3], "min_samples_leaf": [1, 3], "bootstrap": [False], "n_estimators" :[100], "criterion": ["gini"]}
sur_dtc_parmas = {'criterion':['gini', 'entropy'], 'splitter':['best', 'random'],'min_samples_split':[2, 3]}
sur_abc_params = {"base_estimator__criterion" : ["gini", "entropy"], "base_estimator__splitter" :   ["best", "random"], "algorithm" : ["SAMME","SAMME.R"], "n_estimators" :[1,2], "learning_rate":  [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3,1.5]}
sur_etc_params = {"max_depth": [None], "max_features": [1, 3, 10], "min_samples_split": [2, 3, 10], "min_samples_leaf": [1, 3, 10], "bootstrap": [False], "n_estimators" :[100,300], "criterion": ["gini"]}
sur_svc_params = {'kernel': ['rbf'], 'gamma': [ 0.001, 0.01, 0.1, 1], 'C': [1, 10, 50, 100,200,300, 1000]}

# create model and param dictionaries
sur_dict['gbc'] = {'model':sur_gbc_mod, 'params':sur_gbc_params}
sur_dict['rfc'] = {'model':sur_rfc_mod, 'params':sur_rfc_params}
sur_dict['dtc'] = {'model':sur_dtc_mod, 'params':sur_dtc_parmas}
sur_dict['abc'] = {'model':sur_abc_mod, 'params':sur_abc_params}
sur_dict['etc'] = {'model':sur_etc_mod, 'params':sur_etc_params}
sur_dict['svc'] = {'model':sur_svc_mod, 'params':sur_svc_params}