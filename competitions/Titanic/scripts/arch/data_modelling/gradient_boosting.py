# -*- coding: utf-8 -*-
"""
Created on Sat Mar  3 10:04:55 2018

@author: oisin_000
"""

"""
###############################################################################
## Preliminaries ##############################################################
###############################################################################
"""

#-- load libraries --#

print('loading libraries')

# os will be used for set working directory
import os

# pandas will be used for data manipulation
import pandas as pd

# sklearn will be used for the modelling and classification
from sklearn import metrics
from sklearn import model_selection
from sklearn import ensemble

#-- set working directory --#

print('setting working directory')

# get current working directory
os.getcwd()

# set working directory
os.chdir('C:\\Users\\oisin_000\\Documents\\Kaggle\\Titanic Competition')

#-- import data --#

print('loading data')

# load training set
train = pd.read_csv('train.csv')

# load testing set
test = pd.read_csv('test.csv')

# load clean training set
clean_train = pd.read_csv('./python/clean_train.csv')

# load clean testing set
clean_test = pd.read_csv('./python/clean_test.csv')

#-- split the data --#

print('splitting the data')

# subset the training predictors
X = clean_train.drop(labels = 'Survived', axis = 1)

# subset the training response
y = clean_train.Survived

# subset the test set
X_test = clean_test.drop(labels = 'Survived', axis = 1)

#--split into train and validsets --#

# split the data into training and validation sets
X_train, X_valid, y_train, y_valid = model_selection.train_test_split(X, y, 
                                                                      test_size = 0.3,
                                                                      random_state = 123)

"""
###############################################################################
## Gradient Boosting ##########################################################
###############################################################################
"""

#-- Base Model --##############################################################

# initiate knn
gbm = ensemble.GradientBoostingClassifier(random_state = 123)

# save the parametre features to tune as a dictionary
params = {'loss':['deviance', 'exponential'],
          'learning_rate':[1.0, 0.9, 0.8],
          'n_estimators':[50, 100, 200],
          'max_depth':[3, 6, 12],
          'max_features':[None, 'sqrt', 'log2'],
          'presort':[True]}

# initate the tuning procedure, optimise on accuracy
tunegbm = model_selection.GridSearchCV(estimator = gbm,
                                       param_grid = params,
                                       scoring = 'accuracy'
                                       )

# tune the model
tunegbm.fit(X_train, y_train)

# extract the best score
tunegbm.best_score_

# extract the best estimator
tunegbm.best_estimator_

# extract the best parameters
tunegbm.best_params_

# explicitly intiate the tuned model
tunegbm = ensemble.GradientBoostingClassifier(loss = 'exponential',
                                              learning_rate = 0.8,
                                              max_depth = 3,
                                              max_features = 'sqrt',
                                              n_estimators = 50,
                                              presort = True)

# cross validate the base model
model_selection.cross_val_score(estimator = tunegbm,
                                X = X_train,
                                y = y_train,
                                scoring = 'accuracy',
                                cv = 10)

# fit the tuned model
tunegbm.fit(X_train, y_train)

# classify the validation set
y_pred = tunegbm.predict(X_valid)

# genrate the accuracy metric
metrics.accuracy_score(y_valid, y_pred)
# 0.85820895522388063

#-- Bagging Model -############################################################

# intiate the base model
tunegbm = ensemble.GradientBoostingClassifier(loss = 'exponential',
                                              learning_rate = 0.8,
                                              max_depth = 3,
                                              max_features = 'sqrt',
                                              n_estimators = 50,
                                              presort = True)

# initiate bag model
baggbm = ensemble.BaggingClassifier(base_estimator = tunegbm)

# save the parametre features to tune as a dictionary
params = {'n_estimators':[50, 100, 200, 400],
          'max_samples':[1.0, 0.9, 0.8, 0.7, 0.6],
          'max_features':[1.0, 0.8, 0.6, 0.4, 0.2],
          'random_state':[123]}

# initate the tuning procedure, optimise on accuracy
tunebaggbm = model_selection.GridSearchCV(estimator = baggbm,
                                         param_grid = params,
                                         scoring = 'accuracy')

# tune the model
tunebaggbm.fit(X_train, y_train)

# extract the best score
tunebaggbm.best_score_

# extract the best estimator
tunebaggbm.best_estimator_

# extract the best parameters
tunebaggbm.best_params_

# explicitly intiate the tuned bag model
tunebaggbm = ensemble.BaggingClassifier(base_estimator = tunegbm,
                                        max_features = 0.2,
                                        max_samples = 0.7,
                                        n_estimators = 50,
                                        random_state = 123)

# cross validate the bag model
model_selection.cross_val_score(estimator = tunebaggbm,
                                X = X_train,
                                y = y_train,
                                scoring = 'accuracy',
                                cv = 10)

# fit the tuned bag model
tunebaggbm.fit(X_train, y_train)

# classify the validation set
y_pred = tunebaggbm.predict(X_valid)

# genrate the accuracy metric
metrics.accuracy_score(y_valid, y_pred)
# 0.86567164179104472

#-- Final Model & Predictions --###############################################

# fit the final model
tunebaggbm.fit(X, y)

# predict the test set
Survived = tunebaggbm.predict(X_test)

# create passenger id
PassengerId = range(892, 1310)

# save the test predictions as a dataframe
test_df = pd.DataFrame({"PassengerId":PassengerId,
                        "Survived":Survived})

# convert Survived to an integer
test_df.Survived = test_df.Survived.astype(int)

# write to a csv file
test_df.to_csv("./python/gbm_preds.csv",
               index = False)
