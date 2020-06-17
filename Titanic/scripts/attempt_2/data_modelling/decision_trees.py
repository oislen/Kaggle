# -*- coding: utf-8 -*-
"""
Created on Fri Mar  2 14:04:33 2018

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
from sklearn import tree

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
## Decision Tree Model ########################################################
###############################################################################
"""

#-- Base Model --##############################################################

# initiate decision tree
dt = tree.DecisionTreeClassifier()

# save the parametre features to tune as a dictionary
params = {'criterion':['gini', 'entropy'],
          'splitter':['best', 'random'],
          'max_depth':[None, 1, 2, 3, 4, 5],
          'min_samples_split':[2, 3, 4, 5, 6],
          'min_samples_leaf':[1, 2, 3, 4, 5],
          'max_features':[10, 25, 50, 75, 100, 150, 200, 300, 400],
          'random_state':[123],
          'presort':[True]}

# initate the tuning procedure, optimise on accuracy
tunedt = model_selection.GridSearchCV(estimator = dt,
                                      param_grid = params,
                                      scoring = 'accuracy',
                                      n_jobs = 2)

# tune the model
tunedt.fit(X_train, y_train)

# extract the best score
tunedt.best_score_

# extract the best estimator
tunedt.best_estimator_

# extract the best parameters
tunedt.best_params_

# explicitly intiate the tuned model
tunedt = tree.DecisionTreeClassifier(criterion = 'entropy',
                                 max_depth = 3,
                                     max_features = 100,
                                     min_samples_leaf = 1,
                                     min_samples_split = 2,
                                     presort = True,
                                     random_state = 123,
                                     splitter = 'best')

# cross validate the base model
model_selection.cross_val_score(estimator  = tunedt,
                                X = X_train,
                                y = y_train,
                                scoring = 'accuracy',
                                cv = 10,
                                n_jobs = -1)

# fit the tuned model
tunedt.fit(X_train, y_train)

# classify the validation set
y_pred = dt.predict(X_valid)

# genrate the accuracy metric
metrics.accuracy_score(y_valid, y_pred)
# 0.78358208955223885

#-- AdaBoosted Model --########################################################

# intiate the base model
tunedt = tree.DecisionTreeClassifier(criterion = 'entropy',
                                 max_depth = 3,
                                     max_features = 100,
                                     min_samples_leaf = 1,
                                     min_samples_split = 2,
                                     presort = True,
                                     random_state = 123,
                                     splitter = 'best')

# initiate boost model
boostdt = ensemble.AdaBoostClassifier(base_estimator = tunedt)

# save the parametre features to tune as a dictionary
params = {'n_estimators':[10, 25, 50, 100, 150, 200],
          'learning_rate':[1.0, 0.9, 0.8, 0.7, 0.6],
          'algorithm':['SAMME', 'SAMME.R'],
          'random_state':[123]}

# initate the tuning procedure, optimise on accuracy
tuneboostdt = model_selection.GridSearchCV(estimator = boostdt,
                                           param_grid = params,
                                           scoring = 'accuracy',
                                           n_jobs = 2)

# tune the model
tuneboostdt.fit(X_train, y_train)

# extract the best score
tuneboostdt.best_score_

# extract the best estimator
tuneboostdt.best_estimator_

# extract the best parameters
tuneboostdt.best_params_

# explicitly intiate the tuned model
tuneboostdt = ensemble.AdaBoostClassifier(base_estimator = dt,
                                          algorithm = 'SAMME.R',
                                          learning_rate = 0.7,
                                          n_estimators = 25,
                                          random_state = 123)

# cross validate the base model
model_selection.cross_val_score(estimator = tuneboostdt,
                                X = X_train,
                                y = y_train,
                                scoring = 'accuracy',
                                cv = 10,
                                n_jobs = -1)

# fit the tuned model
tuneboostdt.fit(X_train, y_train)

# classify the validation set
y_pred = tuneboostdt.predict(X_valid)

# genrate the accuracy metric
metrics.accuracy_score(y_valid, y_pred)
# 0.83582089552238803

#-- Bagging Model -############################################################

# intiate the base model
tunedt = tree.DecisionTreeClassifier(criterion = 'entropy',
                                 max_depth = 3,
                                     max_features = 100,
                                     min_samples_leaf = 1,
                                     min_samples_split = 2,
                                     presort = True,
                                     random_state = 123,
                                     splitter = 'best')

# initiate bag model
bagdt = ensemble.BaggingClassifier(base_estimator = tunedt)

# save the parametre features to tune as a dictionary
params = {'n_estimators':[10, 50, 100, 200, 400, 800],
          'max_samples':[1.0, 0.9, 0.8, 0.7, 0.6],
          'max_features':[1.0, 0.8, 0.6, 0.4, 0.2],
          'random_state':[123]}

# initate the tuning procedure, optimise on accuracy
tunebagdt = model_selection.GridSearchCV(estimator = bagdt,
                                         param_grid = params,
                                         scoring = 'accuracy',
                                         n_jobs = 2)

# tune the model
tunebagdt.fit(X_train, y_train)

# extract the best score
tunebagdt.best_score_

# extract the best estimator
tunebagdt.best_estimator_

# extract the best parameters
tunebagdt.best_params_

# explicitly intiate the tuned bag model
tunebagdt = ensemble.BaggingClassifier(base_estimator = tunedt,
                                       max_features = 0.4,
                                       max_samples = 0.8,
                                       n_estimators = 10,
                                       n_jobs = -1,
                                       random_state = 123)

# cross validate the bag model
model_selection.cross_val_score(estimator = tunebagdt,
                                X = X_train,
                                y = y_train,
                                scoring = 'accuracy',
                                cv = 10,
                                n_jobs = -1)

# fit the tuned bag model
tunebagdt.fit(X_train, y_train)

# classify the validation set
y_pred = tunebagdt.predict(X_valid)

# genrate the accuracy metric
metrics.accuracy_score(y_valid, y_pred)
# 0.82462686567164178

#-- Bagged and Boosted Model --################################################

# intiate the base model
tunedt = tree.DecisionTreeClassifier(criterion = 'entropy',
                                 max_depth = 3,
                                     max_features = 100,
                                     min_samples_leaf = 1,
                                     min_samples_split = 2,
                                     presort = True,
                                     random_state = 123,
                                     splitter = 'best')

# intiate the boost model
tuneboostdt = ensemble.AdaBoostClassifier(base_estimator = tunedt,
                                          algorithm = 'SAMME.R',
                                          learning_rate = 0.7,
                                          n_estimators = 25,
                                          random_state = 123)

# intiate the bag model
tunebagdt = ensemble.BaggingClassifier(base_estimator = tuneboostdt,
                                       max_features = 0.4,
                                       max_samples = 0.8,
                                       n_estimators = 10,
                                       n_jobs = -1,
                                       random_state = 123)

# fit the tuned bag model
tunebagdt.fit(X_train, y_train)

# classify the validation set
y_pred = tunebagdt.predict(X_valid)

# genrate the accuracy metric
metrics.accuracy_score(y_valid, y_pred)
# 0.86940298507462688

#-- Final Model & Predictions --###############################################

# fit the final model
tunebagdt.fit(X, y)

# predict the test set
Survived = tunebagdt.predict(X_test)

# create passenger id
PassengerId = range(892, 1310)

# save the test predictions as a dataframe
test_df = pd.DataFrame({"PassengerId":PassengerId,
                        "Survived":Survived})

# convert Survived to an integer
test_df.Survived = test_df.Survived.astype(int)

# write to a csv file
test_df.to_csv("./python/preds.csv",
               index = False)
