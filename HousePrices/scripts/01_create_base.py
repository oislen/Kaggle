# -*- coding: utf-8 -*-
"""
Created on Sun Nov 18 12:23:52 2018

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
input_dir = 'C:\\Users\\User\\Documents\\Kaggle\\HousePrices\\data\\'
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

# create a 'SalePrice' column in test
test['SalePrice'] = np.nan

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

# create logSalePrice
base['logSalePrice'] = np.log10(base['SalePrice'])

"""
########################
#-- Output Base File --#
########################
"""

print('Outputting base file ...')

# define the output location and filename
output_filename = 'base.csv'

# output the dataset
base.to_csv(input_dir + output_filename,
               sep = '|',
               encoding = 'utf-8',
               header = True,
               index = False
               )

