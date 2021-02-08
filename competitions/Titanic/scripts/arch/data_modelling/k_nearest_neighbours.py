# -*- coding: utf-8 -*-
"""
Created on Fri Mar  2 14:36:48 2018

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
from sklearn import neighbors

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
## K-Nearest Neighbours #######################################################
###############################################################################
"""

#-- Base Model --##############################################################

# initiate knn
knn = neighbors.KNeighborsClassifier()

# save the parametre features to tune as a dictionary
params = {'n_neighbors':[1, 3, 5, 7, 9, 11, 13],
          'weights':['uniform', 'distance'],
          'algorithm':['auto'],
          'p':[1, 2, 3, 4, 5]}

# initate the tuning procedure, optimise on accuracy
tuneknn = model_selection.GridSearchCV(estimator = knn,
                                       param_grid = params,
                                       scoring = 'accuracy')

# tune the model
tuneknn.fit(X_train, y_train)

# extract the best score
tuneknn.best_score_

# extract the best estimator
tuneknn.best_estimator_

# extract the best parameters
tuneknn.best_params_

# explicitly intiate the tuned model
tuneknn = neighbors.KNeighborsClassifier(n_neighbors = 5,
                                         weights = 'uniform',
                                         algorithm = 'auto',
                                         p = 1)

# cross validate the base model
model_selection.cross_val_score(estimator  = tuneknn,
                                X = X_train,
                                y = y_train,
                                scoring = 'accuracy',
                                cv = 10)

# fit the tuned model
tuneknn.fit(X_train, y_train)

# classify the validation set
y_pred = tuneknn.predict(X_valid)

# genrate the accuracy metric
metrics.accuracy_score(y_valid, y_pred)
# 0.68656716417910446


#-- AdaBoosted Model --########################################################

# AdaBoost does not suppot knn

#-- Bagging Model -############################################################

# intiate the base model
tuneknn = neighbors.KNeighborsClassifier(n_neighbors = 5,
                                         weights = 'uniform',
                                         algorithm = 'auto',
                                         p = 1)

# initiate bag model
bagknn = ensemble.BaggingClassifier(base_estimator = tuneknn)

# save the parametre features to tune as a dictionary
params = {'n_estimators':[10, 50, 100, 200, 400, 800],
          'max_samples':[1.0, 0.9, 0.8, 0.7, 0.6],
          'max_features':[1.0, 0.8, 0.6, 0.4, 0.2],
          'random_state':[123]}

# initate the tuning procedure, optimise on accuracy
tunebagknn = model_selection.GridSearchCV(estimator = bagknn,
                                          param_grid = params,
                                          scoring = 'accuracy')

# tune the model
tunebagknn.fit(X_train, y_train)

# extract the best score
tunebagknn.best_score_

# extract the best estimator
tunebagknn.best_estimator_

# extract the best parameters
tunebagknn.best_params_

# explicitly intiate the tuned bag model
tunebagknn = ensemble.BaggingClassifier(base_estimator = tuneknn,
                                        max_features = 0.2,
                                        max_samples = 0.9,
                                        n_estimators = 200,
                                        random_state = 123)

# cross validate the bag model
model_selection.cross_val_score(estimator = tunebagknn,
                                X = X_train,
                                y = y_train,
                                scoring = 'accuracy',
                                cv = 10)

# fit the tuned bag model
tunebagknn.fit(X_train, y_train)

# classify the validation set
y_pred = tunebagknn.predict(X_valid)

# genrate the accuracy metric
metrics.accuracy_score(y_valid, y_pred)
# 0.84701492537313428

#-- Final Model & Predictions --###############################################

# fit the final model
tunebagknn.fit(X, y)

# predict the test set
Survived = tunebagknn.predict(X_test)

# create passenger id
PassengerId = range(892, 1310)

# save the test predictions as a dataframe
test_df = pd.DataFrame({"PassengerId":PassengerId,
                        "Survived":Survived})

# convert Survived to an integer
test_df.Survived = test_df.Survived.astype(int)

# write to a csv file
test_df.to_csv("./python/knn_preds.csv",
               index = False)
