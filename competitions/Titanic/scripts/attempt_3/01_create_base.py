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

# load in data
input_dir = 'C:\\Users\\User\\Documents\\Kaggle\\Titanic\\data\\'
train_name = 'train.csv'
test_name = 'test.csv'
train = pd.read_csv(input_dir + train_name, sep = ',')
test = pd.read_csv(input_dir + test_name, sep = ',')

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

# define the output location and filename
output_dir = 'C:\\Users\\User\\Documents\\Kaggle\\Titanic\\data\\attempt_3\\'
output_filename = 'base.csv'

# output the dataset
base.to_csv(output_dir + output_filename,
               sep = '|',
               encoding = 'utf-8',
               header = True,
               index = False
               )
