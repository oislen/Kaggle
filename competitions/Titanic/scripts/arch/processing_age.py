# -*- coding: utf-8 -*-
"""
Created on Sat Mar  3 14:34:10 2018

@author: oisin_000
"""
"""
###############################################################################
## Preliminaries ##############################################################
###############################################################################

This script builds a prediction model for the ages of passengers
the the main data processing script should be run up untill the point of age
then this script should be run with the results being piped into the next stages
Note could implement two models one with survived and one without survived
"""

#-- load libraries --#

print('loading libraries')

# os will be used for set working directory
import os

# pandas will be used for data manipulation
import pandas as pd

# numpy will be used for functions
import numpy as np

# sklearn will be used for the modelling and classification
from sklearn import feature_selection
from sklearn import metrics
from sklearn import model_selection
from sklearn import ensemble
from sklearn import preprocessing
from sklearn import tree

#-- set working directory --#

print('setting working directory')

# get current working directory
os.getcwd()

# set working directory
os.chdir('C:\\Users\\oisin_000\\Documents\\Kaggle\\Titanic Competition\\python')

#-- import data --#

print('loading data')

# load the dataset in
data = pd.read_csv('data_pII.csv')

"""
###############################################################################
#-- Age --#####################################################################
###############################################################################
"""

print('processing age')

#-- impute with classification techniques --###################################

# how many missing values
data.Age.isnull().sum()
# extract the data for imputing age
age_data = data
# drop the survived variable
age_data = age_data.drop(labels = 'Survived', axis = 1)

"""
###############################################################################
## Dummy Encode Categorical Variables #########################################
###############################################################################
"""

print('dummy encoding the categorical varaibles')

# dummy encode the categorical variables
age_data = pd.get_dummies(age_data)


# extract target age
age_data_y = age_data.Age
# extract predictors
age_data_X = age_data.drop(labels = 'Age', axis = 1)

"""
###############################################################################
#-- derive terms --############################################################
###############################################################################
"""

print('deriving interaction and polynomial terms')

# initiate interaction and polynomials
poly = preprocessing.PolynomialFeatures()
# fit poly terms
poly_data = poly.fit_transform(age_data_X)
# save column names
col_names = poly.get_feature_names(age_data_X.columns)
# turn poly df 
age_data_X = pd.DataFrame(poly_data, columns = col_names)

"""
###############################################################################
#-- select best terms --#######################################################
###############################################################################
"""

print('select the top 50 variables')

# need to use non null data
# extract the train set
age_train_X = age_data_X[age_data.Age.notnull()]
age_train_y = age_data_y[age_data_y.notnull()]
# intiate select top 100 attributes
selector = feature_selection.SelectKBest(score_func = feature_selection.f_regression, 
                                         k = 500)
# select the attributes
select_data = selector.fit_transform(age_train_X, age_train_y)
# get colnames
col_names = age_train_X.columns[selector.get_support(indices = True)]
# save features
age_train_X = pd.DataFrame(select_data, columns = col_names)
# save the column names 
col_names = age_train_X.columns
# extract the best columns from the null data
age_data_X = age_data_X.loc[:, col_names]
# extract the null data
age_testXXX = age_data_X[age_data_y.isnull()]

"""
###############################################################################
## Scale the data #############################################################
###############################################################################
"""


"""
###############################################################################
## Data Modelling #############################################################
###############################################################################
"""

#--split into train and validsets --#

# split the data into training and test sets
age_train_XX, age_test_XX, age_train_yy, age_test_yy = model_selection.train_test_split(age_train_X, age_train_y, 
                                                                                        test_size = 0.3,
                                                                                        random_state = 123)

#-- base data --##############################################################

# initiate regression tree
md = tree.DecisionTreeRegressor(random_state = 123) # 104.43462
#md = svm.SVR()
#md = neighbors.KNeighborsRegressor() # 143.56
#md = linear_model.ElasticNet() # 141.576087
# initiate boost model
# boostdt = ensemble.AdaBoostRegressor(base_estimator= dt,
#                                      n_estimators = 100)


#-- initiate ensemble model --#################################################

bagdt = ensemble.BaggingRegressor(base_estimator = md,
                                  n_estimators = 1000,
                                  max_features = 60,
                                  random_state = 123)
# fit the decision tree
bagdt.fit(age_train_XX, age_train_yy)
# predict a validation set for age
age_valid = bagdt.predict(age_test_XX)
# calculate the mean squared error for the validation set
metrics.mean_squared_error(age_test_yy, age_valid)
# predict the test set
age_pred = bagdt.predict(age_testXXX)
# assign the imputed ages to the age variable
data.Age[data.Age.isnull()] = age_pred
# how many missing values
data.Age.isnull().sum()

# delete excess data
del age_data, age_data_X, age_data_y, age_pred, age_testXXX, age_test_XX, 
del age_test_yy, age_train_X, age_train_XX, age_train_y, age_train_yy, 
del age_valid, poly_data, select_data

