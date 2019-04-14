# -*- coding: utf-8 -*-
"""
Created on Sun Nov 18 22:11:39 2018

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
import statsmodels.api as sm

# load cusotm functions
import sys
sys.path.append('C:/Users/User/Documents/Data_Analytics/Python/value_analysis')
import value_analysis as va

# load in data
input_dir = 'C:\\Users\\User\\Documents\\Kaggle\\HousePrices\\data\\'
clean_name = 'clean.csv'
clean = pd.read_csv(input_dir + clean_name, 
                    sep = '|'
                    )

"""
##########################
#-- Feature Importance --#
##########################
"""

print('Performing feature importance ...')

test = clean[(clean['Dataset'] == 'test')]
train = clean[(clean['Dataset'] == 'train')]

# randomly split the dataset
X_valid, X_train, y_valid, y_train = va.train_test_split_sample(dataset = train,
                                                                y = ['logSalePrice'],
                                                                X = train.columns.drop(['Id', 'logSalePrice', 'Dataset']),
                                                                train_size = 0.8,
                                                                test_size = 0.2
                                                                )

# calculate feature importance
feat_imp = va.GLM_simple_analysis(endog = y_train,
                                  exog = X_train,
                                  family = sm.families.Gaussian()
                                  )

feat_imp.head(30)
