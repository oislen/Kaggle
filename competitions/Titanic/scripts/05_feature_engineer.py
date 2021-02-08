# -*- coding: utf-8 -*-
"""
Created on Sun Nov  4 21:46:26 2018

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
from sklearn import ensemble
import cons

# load cusotm functions
import value_analysis as va

# load in data
base = pd.read_csv(cons.base_clean_2_data_fpath, 
                   sep = '|'
                   )

"""
########################
#-- Feature Engineer --#
########################
"""

print('Deriving interaction terms ...')

# extract the columns to create interaction terms for
int_cols = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'FamSize']

# create interaction terms
int_data = va.derive_variables(dataset = base,
                               attr = int_cols,
                               var_type = 'interaction'
                               )

# create the engineered data by concatenating the base data with the interaction data
engin = pd.concat(objs = [base, int_data],
                  axis = 1
                  )

"""
##########################
#-- Standardising Data --#
##########################
"""

print('Standardising data ...')

# define the columns to standardise
stand_cols = int_data.columns.tolist()

# standardise data to interval [0, 1]
stand = va.standardise_variables(dataset = int_data,
                                 attr = stand_cols,
                                 stand_type = 'range',
                                 stand_range = [0, 1]
                                 )

# update the processed data
engin[stand_cols] = stand

"""
##########################
#-- Feature Importance --#
##########################
"""

print('Performing feature importance ...')

# randomly split the dataset
X_valid, X_train, y_valid, y_train = va.train_test_split_sample(dataset = engin,
                                                                y = ['Survived'],
                                                                X = engin.columns.drop(['PassengerId', 'Dataset', 'Survived', 'Ticket_Number']),
                                                                train_size = 0.8,
                                                                test_size = 0.2,
                                                                random_split = True,
                                                                sample_target = 'Survived',
                                                                sample_type = 'over'
                                                                )

# create a gbm model
gbm = ensemble.GradientBoostingClassifier(random_state = 123)

# determine the feature importance
feat_imp = va.tree_feat_imp(model = gbm,
                            y_train = y_train,
                            X_train = X_train
                            )

# extract out the important features 
best_feat = feat_imp.index[feat_imp['Importance'] > 1].tolist()

# add in additional variables to enable the interaction effects
add_vars = ['Pclass', 'Embarked', 'FamSize', 'SibSp']

# subset the best data
best_data = engin[best_feat + add_vars]

# create the final dataset
final_data = pd.concat(objs = [base[['PassengerId', 'Dataset', 'Survived']], best_data],
                       axis = 1
                       )

"""
##############
#-- Output --#
##############
"""

print('Outputting data ...')

# output the dataset
final_data.to_csv(cons.base_engin_data_fpath,
                  sep = '|',
                  encoding = 'utf-8',
                  header = True,
                  index = False
                  )