# -*- coding: utf-8 -*-
"""
Created on Fri Mar  2 17:22:20 2018

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
from sklearn import neural_network

#-- set working directory --#

print('setting working directory')

# get current working directory
os.getcwd()

# set working directory
os.chdir('C:\\Users\\oisin_000\\Documents\\Kaggle\\Titanic Competition\\python')

#-- import data --#

print('loading data')

# load clean training set
clean_train = pd.read_csv('./data/clean_train.csv')

# load clean testing set
clean_test = pd.read_csv('./data/clean_test.csv')

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
## Neural Networks ############################################################
###############################################################################
"""

"""
MLP Neural Network features backward propagation algorithm.
Input layer has k number of nodes as columns
Output layer has one node for the target variable
Hidden layer has h nodes where k >= h >= 1
"""

#-- Base Model --##############################################################

# initiate knn
nn = neural_network.MLPClassifier(random_state = 123)

# save the parametre features to tune as a dictionary
params = {'hidden_layer_sizes':[(10,), (20,), (30,), (40,), (50,)],
          'activation':['logistic', 'tanh', 'relu'],
          'learning_rate':['constant', 'invscaling', 'adaptive'],
          'solver':['lbfgs', 'sgd', 'adam'],
          'max_iter':[500]}

# initate the tuning procedure, optimise on accuracy
tunenn = model_selection.GridSearchCV(estimator = nn,
                                      param_grid = params,
                                      scoring = 'accuracy')

# tune the model
tunenn.fit(X_train, y_train)

# extract the best score
tunenn.best_score_

# extract the best estimator
tunenn.best_estimator_

# extract the best parameters
tunenn.best_params_

# explicitly intiate the tuned model
tunenn = neural_network.MLPClassifier(hidden_layer_sizes = (20,),
                                      activation = 'tanh',
                                      learning_rate = 'constant',
                                      solver = 'lbfgs',
                                      max_iter = 500,
                                      random_state = 123)

# cross validate the base model
model_selection.cross_val_score(estimator = tunenn,
                                X = X_train,
                                y = y_train,
                                scoring = 'accuracy',
                                cv = 10)

# fit the tuned model
tunenn.fit(X_train, y_train)

# classify the validation set
y_pred = tunenn.predict(X_valid)

# genrate the accuracy metric
metrics.accuracy_score(y_valid, y_pred)
# 0.77272727272727271

#-- Bagging Model -############################################################

# intiate the base model
tunenn = neural_network.MLPClassifier(hidden_layer_sizes = (20,),
                                      activation = 'tanh',
                                      learning_rate = 'constant',
                                      solver = 'lbfgs',
                                      max_iter = 500,
                                      random_state = 123)

# initiate bag model
bagnn = ensemble.BaggingClassifier(base_estimator = tunenn,
                                   random_state = 123)

# save the parametre features to tune as a dictionary
params = {'n_estimators':[10, 20, 40, 80, 160],
          'max_samples':[1.0, 0.9, 0.8, 0.7, 0.6],
          'max_features':[1.0, 0.9, 0.8, 0.7, 0.6]}

# initate the tuning procedure, optimise on accuracy
tunebagnn = model_selection.GridSearchCV(estimator = bagnn,
                                         param_grid = params,
                                         scoring = 'accuracy')

# tune the model
tunebagnn.fit(X_train, y_train)

# extract the best score
tunebagnn.best_score_

# extract the best estimator
tunebagnn.best_estimator_

# extract the best parameters
tunebagnn.best_params_

# explicitly intiate the tuned bag model
tunebagnn = ensemble.BaggingClassifier(base_estimator = tunenn,
                                       max_features = 0.6,
                                       max_samples = 1.0,
                                       n_estimators = 40,
                                       random_state = 123)

# cross validate the bag model
model_selection.cross_val_score(estimator = tunebagnn,
                                X = X_train,
                                y = y_train,
                                scoring = 'accuracy',
                                cv = 100)

# fit the tuned bag model
tunebagnn.fit(X_train, y_train)

# classify the validation set
y_pred = tunebagnn.predict(X_valid)

# genrate the accuracy metric
metrics.accuracy_score(y_valid, y_pred)
# 0.81818181818181823

#-- Final Model & Predictions --###############################################

# fit the final model
tunebagnn.fit(X, y)

# predict the test set
Survived = tunebagnn.predict(X_test)

# create passenger id
PassengerId = range(892, 1310)

# save the test predictions as a dataframe
test_df = pd.DataFrame({"PassengerId":PassengerId,
                        "Survived":Survived})

# convert Survived to an integer
test_df.Survived = test_df.Survived.astype(int)

# write to a csv file
test_df.to_csv("./predictions/nn_preds.csv",
               index = False)
