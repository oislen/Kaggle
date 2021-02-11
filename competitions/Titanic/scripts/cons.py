# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 17:20:45 2021

@author: oislen
"""

# load libraries
import os
import sys
import pandas as pd

pd.set_option('display.max_columns', 20)

#-- Filepath Constants --#

# set programme constants
comp_name = 'titanic'
download_data = True
unzip_data = True
del_zip = True

# set directories
git_dir = 'C:\\Users\\User\\Documents\\GitHub'
root_dir = os.path.join(git_dir, 'Kaggle')
va_dir = os.path.join(git_dir, 'value_analysis')
utilities_dir = os.path.join(root_dir, 'utilities')
titanic_comp_dir = os.path.join(root_dir, 'competitions\\Titanic')
scripts_dir = os.path.join(titanic_comp_dir, 'scripts')
data_dir = os.path.join(titanic_comp_dir, 'data')
report_dir = os.path.join(titanic_comp_dir, 'report')
univar_stats_dir = os.path.join(report_dir, 'univariate_analysis\\descriptive_stats')
bivar_assoc_dir = os.path.join(report_dir, 'bivariate_analysis\\association_tests')
bivar_corr_dir = os.path.join(report_dir, 'bivariate_analysis\\correlation_tests')
bivar_gains_dir = os.path.join(report_dir, 'bivariate_analysis\\measure_gains')

# define filenames
zip_data_fname = '{}.zip'.format(comp_name)
sample_sub_data_fname = 'gender_submission.csv'
test_data_fname = 'test.csv'
train_data_fname = 'train.csv'
base_data_fname = 'base.csv'
base_clean_data_fname = 'base_clean.csv'
base_clean_2_data_fname = 'base_clean_2.csv'
base_engin_fname = 'base_engin.csv'
pred_data_fname = 'preds.csv'

# create file paths
zip_data_fpath = os.path.join(data_dir, zip_data_fname)
sample_sub_data_fpath = os.path.join(data_dir, sample_sub_data_fname)
test_data_fpath = os.path.join(data_dir, test_data_fname)
train_data_fpath = os.path.join(data_dir, train_data_fname)
base_data_fpath = os.path.join(data_dir, base_data_fname)
base_clean_data_fpath = os.path.join(data_dir, base_clean_data_fname)
base_clean_2_data_fpath = os.path.join(data_dir, base_clean_2_data_fname)
base_engin_data_fpath = os.path.join(data_dir, base_engin_fname)
pred_data_fpath = os.path.join(data_dir, pred_data_fname)

# append utilities directory to path
for p in [utilities_dir, va_dir]:
    sys.path.append(p)
    
#-- Cleaning Constants --#

# set base columns
sub_cols = ['PassengerId', 'Survived', 'Pclass', 'Age',
            'SibSp', 'Parch', 'FamSize', 'Fare', 'Alone', 
            'Mr', 'Mrs', 'Ms', 'Priv', 'Male',
            'Embarked_Ord', 'Dataset']

# the title of a person indicates the person status
title_map = {'Mr':'Mr', 
             'Miss':'Ms', 
             'Mrs':'Mrs', 
             'Master':'Master',
             'Dr':'Dr',
             'Rev':'Rev',
             'Col':'Col',
             'Major':'Major',
             'Ms':'Ms',
             'Mlle':'Ms',
             'Capt':'Capt',
             'Don':'Mr',
             'Jonkheer':'Jonkheer',
             'Sir':'Sir',
             'Lady':'Lady',
             'Mme':'Mrs',
             'Dona':'Mrs',
             'theCountess':'Countess'
             }

    
# create a map for the title values
priv_map = {'Mr':'Mr', 
            'Ms':'Ms', 
            'Mrs':'Mrs', 
            'Master':'Priv',
            'Dr':'Priv',
            'Rev':'Priv',
            'Col':'Priv',
            'Major':'Priv',
            'Capt':'Priv',
            'Jonkheer':'Priv',
            'Sir':'Priv',
            'Lady':'Priv',
            'Countess':'Priv'
            }

# create ordinal mapping for title
title_ord_map = {'Mr':1,
                 'Ms':2,
                 'Mrs':3,
                 'Priv':4
                 }

# create cabin map
cab_map = {'A':'A', 'B':'B', 'BB':'B', 'BBB':'B', 'BBBB':'B', 'C':'C', 
           'CC':'C', 'CCC':'C', 'D':'D', 'DD':'D', 'E':'E', 'EE':'E',
           'F':'F', 'FG':'F', 'FE':'F', 'G':'G', 'T':'T'
           }

# create embarked map
embarked_map = {'S':1, 'C':2, 'Q':3}

