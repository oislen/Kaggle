# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 09:58:04 2019

@author: oislen
"""

# load in the relevant libraries
import pandas as pd
import numpy as np
from sklearn import linear_model

# load cusotm functions
import sys
va_dir = 'C:/Users/User/Documents/Data_Analytics/Python/value_analysis'
sys.path.append(va_dir)
import value_analysis as va

# load in data
input_dir = 'C:\\Users\\User\\Documents\\GitHub\\Kaggle\\HousePrices\\data\\'
clean_name = 'clean.csv'
clean = pd.read_csv(input_dir + clean_name, 
                    sep = '|'
                    )

# create the output directory
output_dir = input_dir + 'python_predictions\\'

"""
########################
#-- Standardise data --#
########################
"""

print('Standardising data ...')

# define reference columns
ref_cols = ['Id', 'Dataset', 'logSalePrice']

# create the standardisation dataset
stand = pd.DataFrame()
stand[ref_cols] = clean[ref_cols]

# create a list of variables to standardise
stand_cols = clean.columns.drop(ref_cols).tolist()
#stand_cols = ['BsmtFinSF2', 'LowQualFinSF', 'BsmtHalfBath']

# standardise the dataset using the robust scalar method
stand[stand_cols] = va.standardise_variables(dataset = clean,
                                             attr = stand_cols,
                                             stand_type = 'robust'
                                             )

"""
#########################
#-- Split the Dataset --#
#########################
"""

print('Splitting data ...')

# split the datasets
test = clean[(clean['Dataset'] == 'test')]
train = clean[(clean['Dataset'] == 'train')]

# randomly split the dataset
X_valid, X_train, y_valid, y_train = va.train_test_split_sample(dataset = train,
                                                                y = ['logSalePrice'],
                                                                X = train.columns.drop(['Id', 'logSalePrice', 'Dataset']),
                                                                train_size = 0.8,
                                                                test_size = 0.2
                                                                )

"""
########################
#-- Fit Inital LASSO --#
########################
"""

print('Fitting inital model ...')

# define a lasso model
clf = linear_model.Lasso(alpha = 0.0005, 
                         random_state = 1,
                         max_iter  = 10000
                         )

# fit the lasso model
clf.fit(X_train, y_train['logSalePrice'])

# predict for the validation set
y_valid_preds = clf.predict(X_valid)

# generate some metrics
va.metrics(y_obs = np.exp(y_valid['logSalePrice']),
           y_pred = np.exp(y_valid_preds),
           target_type = 'reg'
           )

"""
#######################
#-- Fit Final LASSO --#
#######################
"""

print('Fitting final model ...')

# columns to drop from test set
drop_cols = ['Id', 'Dataset', 'logSalePrice']

# refit a lasso model to the full training data
clf.fit(train.drop(columns = drop_cols), train['logSalePrice'])

# predict for the test set
y_test_preds = np.exp(clf.predict(test.drop(columns = drop_cols)))

"""
#############################
#-- Create the Submission --#
#############################
"""

print('Creating submission file ...')

# create a dataframe for the kaggle submission
sub_df = pd.DataFrame({'Id':test['Id'],
                       'SalePrice':y_test_preds}
                     )


# output the dataset
sub_df.to_csv(output_dir + 'preds1.csv',
              sep = ',',
              encoding = 'utf-8',
              header = True,
              index = False
              )
