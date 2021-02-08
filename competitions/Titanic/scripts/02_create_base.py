# -*- coding: utf-8 -*-
"""
Created on Sat Nov  3 10:42:39 2018

@author: oislen
"""

"""
#####################
#-- Preliminaries --#
#####################
"""

print('Loading in libraries and data ...')

# load in libraries
import pandas as pd
import numpy as np
import cons 

# load in data
train = pd.read_csv(cons.train_data_fpath, sep = ',')
test = pd.read_csv(cons.test_data_fpath, sep = ',')

"""
#########################
#-- Concatenate Files --#
#########################
"""

print('Concatenating files ...')

# note the training set as an extra column, the target survived
train.columns
test.columns

# create a 'Survived' column in test
test['Survived'] = np.nan

# create a dataset indicator column
train['Dataset'] = 'train'
test['Dataset'] = 'test'

# row bind the datasets
base = pd.concat(objs = [train, test], axis = 0, sort = False)

"""
###########################
#-- Feature Engineering --#
###########################
"""

print('Engineering new features ...')

#-- Family Size --#

# create a family size attribute
base['FamSize'] = base['Parch'] + base['SibSp']

#-- Alone --#

# create an alone attribute
base['Alone'] = (base['FamSize'] == 0).astype(int)

"""
########################
#-- Output Base File --#
########################
"""

print('Outputting base file ...')

# output the dataset
base.to_csv(cons.base_data_fpath,
               sep = '|',
               encoding = 'utf-8',
               header = True,
               index = False
               )
