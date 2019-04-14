# -*- coding: utf-8 -*-
"""
Created on Sat Mar  3 11:30:31 2018

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
## Random Forests #############################################################
###############################################################################
"""

#-- Base Model --##############################################################

# initiate knn
rfm = ensemble.RandomForestClassifier(random_state = 123)

# save the parametre features to tune as a dictionary
params = {'n_estimators':[10, 50, 250, 500],
          'criterion':['gini', 'entropy'],
          'max_features':[None, 'sqrt', 'log2']}

# initate the tuning procedure, optimise on accuracy
tunerfm = model_selection.GridSearchCV(estimator = rfm,
                                       param_grid = params,
                                       scoring = 'accuracy')

# tune the model
tunerfm.fit(X_train, y_train)

# extract the best score
tunerfm.best_score_

# extract the best estimator
tunerfm.best_estimator_

# extract the best parameters
tunerfm.best_params_

# explicitly intiate the tuned model
tunerfm = ensemble.RandomForestClassifier(n_estimators = 50,
                                          criterion = 'gini',
                                          max_features = None)

# cross validate the base model
model_selection.cross_val_score(estimator = tunerfm,
                                X = X_train,
                                y = y_train,
                                scoring = 'accuracy',
                                cv = 10)

# fit the tuned model
tunerfm.fit(X_train, y_train)

# classify the validation set
y_pred = tunerfm.predict(X_valid)

# genrate the accuracy metric
metrics.accuracy_score(y_valid, y_pred)
# 0.85447761194029848

#-- Bagging Model -############################################################

# intiate the base model
tunerfm = ensemble.RandomForestClassifier(n_estimators = 50,
                                          criterion = 'gini',
                                          max_features = None)
# initiate bag model
bagrfm = ensemble.BaggingClassifier(base_estimator = tunerfm)

# save the parametre features to tune as a dictionary
params = {'n_estimators':[50, 100, 200, 400],
          'max_samples':[1.0, 0.9, 0.8, 0.7, 0.6],
          'max_features':[1.0, 0.8, 0.6, 0.4, 0.2],
          'random_state':[123]}

# initate the tuning procedure, optimise on accuracy
tunebagrfm = model_selection.GridSearchCV(estimator = bagrfm,
                                          param_grid = params,
                                          scoring = 'accuracy')
 
# tune the model
tunebagrfm.fit(X_train, y_train)

# extract the best score
tunebagrfm.best_score_

# extract the best estimator
tunebagrfm.best_estimator_

# extract the best parameters
tunebagrfm.best_params_

# explicitly intiate the tuned bag model
tunebagrfm = ensemble.BaggingClassifier(base_estimator = tunerfm,
                                        max_features = 0.2,
                                        max_samples = 0.7,
                                        n_estimators = 50,
                                        random_state = 123)

# cross validate the bag model
model_selection.cross_val_score(estimator = tunebagrfm,
                                X = X_train,
                                y = y_train,
                                scoring = 'accuracy',
                                cv = 10)

# fit the tuned bag model
tunebagrfm.fit(X_train, y_train)

# classify the validation set
y_pred = tunebagrfm.predict(X_valid)

# genrate the accuracy metric
metrics.accuracy_score(y_valid, y_pred)
# 0.86567164179104472

#-- Final Model & Predictions --###############################################

# fit the final model
tunebagrfm.fit(X, y)

# predict the test set
Survived = tunebagrfm.predict(X_test)

# create passenger id
PassengerId = range(892, 1310)

# save the test predictions as a dataframe
test_df = pd.DataFrame({"PassengerId":PassengerId,
                        "Survived":Survived})

# convert Survived to an integer
test_df.Survived = test_df.Survived.astype(int)

# write to a csv file
test_df.to_csv("./python/rfm_preds.csv",
               index = False)
