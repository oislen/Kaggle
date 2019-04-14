# -*- coding: utf-8 -*-
"""
Created on Fri Mar  2 16:45:57 2018

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
from sklearn import feature_selection
from sklearn import metrics
from sklearn import model_selection
from sklearn import ensemble
from sklearn import linear_model
from sklearn import preprocessing
from sklearn import naive_bayes
from sklearn import neighbors
from sklearn import neural_network
from sklearn import tree
from sklearn import svm

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
## Naive Bayes ################################################################
###############################################################################
"""

#-- Base Model --##############################################################

# initiate knn
bayes = naive_bayes.MultinomialNB()

# save the parametre features to tune as a dictionary
params = {'alpha':[1.0, 0.8, 0.6, 0.4, 0.2, 0],
          'fit_prior':[True, False]}

# initate the tuning procedure, optimise on accuracy
tunebayes = model_selection.GridSearchCV(estimator = bayes,
                                         param_grid = params,
                                         scoring = 'accuracy')

# tune the model
tunebayes.fit(X_train, y_train)

# extract the best score
tunebayes.best_score_

# extract the best estimator
tunebayes.best_estimator_

# extract the best parameters
tunebayes.best_params_

# explicitly intiate the tuned model
tunebayes = naive_bayes.MultinomialNB(alpha = 0,
                                      fit_prior = True)

# cross validate the base model
model_selection.cross_val_score(estimator = tunebayes,
                                X = X_train,
                                y = y_train,
                                scoring = 'accuracy',
                                cv = 10)

# fit the tuned model
tunebayes.fit(X_train, y_train)

# classify the validation set
y_pred = tunebayes.predict(X_valid)

# genrate the accuracy metric
metrics.accuracy_score(y_valid, y_pred)
# 0.7425373134328358


#-- AdaBoosted Model --########################################################

# AdaBoost does not suppot knn

#-- Bagging Model -############################################################

# intiate the base model
tunebayes = naive_bayes.MultinomialNB(alpha = 0,
                                      fit_prior = True)

# initiate bag model
bagbayes = ensemble.BaggingClassifier(base_estimator = tunebayes)

# save the parametre features to tune as a dictionary
params = {'n_estimators':[10, 50, 100, 200, 400, 800],
          'max_samples':[1.0, 0.9, 0.8, 0.7, 0.6],
          'max_features':[1.0, 0.8, 0.6, 0.4, 0.2],
          'random_state':[123]}

# initate the tuning procedure, optimise on accuracy
tunebagbayes = model_selection.GridSearchCV(estimator = bagbayes,
                                          param_grid = params,
                                          scoring = 'accuracy')

# tune the model
tunebagbayes.fit(X_train, y_train)

# extract the best score
tunebagbayes.best_score_

# extract the best estimator
tunebagbayes.best_estimator_

# extract the best parameters
tunebagbayes.best_params_

# explicitly intiate the tuned bag model
tunebagbayes = ensemble.BaggingClassifier(base_estimator = tunebagbayes,
                                        max_features = 0.2,
                                        max_samples = 10,
                                        n_estimators = 10,
                                        random_state = 123)

# cross validate the bag model
model_selection.cross_val_score(estimator = tunebagbayes,
                                X = X_train,
                                y = y_train,
                                scoring = 'accuracy',
                                cv = 10)

# fit the tuned bag model
tunebagbayes.fit(X_train, y_train)

# classify the validation set
y_pred = tunebagbayes.predict(X_valid)

# genrate the accuracy metric
metrics.accuracy_score(y_valid, y_pred)
# 0.84701492537313428

#-- Final Model & Predictions --###############################################

# fit the final model
tunebagbayes.fit(X, y)

# predict the test set
Survived = tunebagbayes.predict(X_test)

# create passenger id
PassengerId = range(892, 1310)

# save the test predictions as a dataframe
test_df = pd.DataFrame({"PassengerId":PassengerId,
                        "Survived":Survived})

# convert Survived to an integer
test_df.Survived = test_df.Survived.astype(int)

# write to a csv file
test_df.to_csv("./python/bayes_preds.csv",
               index = False)
