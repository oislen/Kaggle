# -*- coding: utf-8 -*-
"""
Created on Sun Nov  4 22:19:35 2018

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

# sklearn will be used for the modelling and classification
from sklearn import ensemble

# load cusotm functions
import sys
sys.path.append('C:/Users/User/Documents/Data_Analytics/Python/value_analysis')
import value_analysis as va

# load in data
input_dir = 'C:\\Users\\User\\Documents\\Kaggle\\Titanic\\data\\attempt_3\\'
base_name = 'base_engin.csv'
base = pd.read_csv(input_dir + base_name, 
                   sep = '|'
                   )

# split the data based on the original dataset
base_train = base[base.Dataset == 'train']
base_test = base[base.Dataset == 'test']

"""
#########################
#-- Classify Survived --#
#########################
"""

print('Classifying for Survived ...')

# randomly split the dataset
X_valid, X_train, y_valid, y_train = va.train_test_split_sample(dataset = base_train,
                                                                y = ['Survived'],
                                                                X = base_train.columns.drop(['PassengerId', 'Dataset', 'Survived']),
                                                                train_size = 0.8,
                                                                test_size = 0.2,
                                                                random_split = True,
                                                                sample_target = 'Survived',
                                                                sample_type = 'over'
                                                                )

# initiate gbm
gbm = ensemble.GradientBoostingClassifier(random_state = 123)

# save the parametre features to tune as a dictionary
params = {'loss':['deviance', 'exponential'],
          'subsample':[1, 0.9, 0.8],
          'learning_rate':[1.0, 0.9, 0.8],
          'min_samples_split':[2, 3, 4],
          'min_samples_leaf':[1, 2, 3],
          'n_estimators':[25, 50, 75],
          'max_depth':[1, 3, 5],
          'max_features':['auto', 'sqrt', 'log2'],
          'presort':[True]
          }

# tune gbm model
mod_tuning = va.tune_hyperparameters(model = gbm, 
                                     params = params, 
                                     X_train = X_train, 
                                     y_train = y_train,
                                     scoring = 'accuracy'
                                     )

# extract the best parameters
best_params = mod_tuning.loc[0, 'params']

# initiate the best model
gbm = ensemble.GradientBoostingClassifier(learning_rate = best_params['learning_rate'],
                                          loss = best_params['loss'],
                                          subsample = best_params['subsample'],
                                          min_samples_split = best_params['min_samples_split'],
                                          min_samples_leaf = best_params['min_samples_leaf'],
                                          max_depth = best_params['max_depth'],
                                          max_features = best_params['max_features'],
                                          n_estimators = best_params['n_estimators'],
                                          presort = best_params['presort'],
                                          random_state = 123
                                          )

# fit the best model
gbm.fit(X_train, 
        y_train.values.ravel()
        )

# classify the validation set
y_valid['Survived_pred'] = gbm.predict(X_valid)

# genrate the regression metrics
with pd.option_context('display.max_columns', None):
    print(va.metrics(y_obs = y_valid['Survived'], 
                     y_pred = y_valid['Survived_pred'], 
                     target_type = 'class'
                     ))

# create a ROC curve
va.vis_roc_curve(obs = 'Survived', preds = 'Survived_pred', dataset = y_valid)

# predict for the test set
base_test['Survived'] = gbm.predict(base_test[X_valid.columns])

"""
##############################
#-- Output Classifications --#
##############################
"""

print('Outputting classifictions ...')

# create the test classification dataset
predictions = pd.DataFrame()
predictions['Survived'] = base_test['Survived'] .astype(int)
predictions['PassengerId'] = base_test.index + 1
predictions = predictions[['PassengerId', 'Survived']]

# define the output location and filename
output_dir = 'C:\\Users\\User\\Documents\\Kaggle\\Titanic\\data\\attempt_3\\'
output_filename = 'preds9.csv'

# output the dataset
predictions.to_csv(output_dir + output_filename,
                   sep = ',',
                   encoding = 'utf-8',
                   header = True,
                   index = False
                   )
