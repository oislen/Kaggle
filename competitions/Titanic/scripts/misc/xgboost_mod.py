# -*- coding: utf-8 -*-
"""
Created on Sun Feb 14 10:48:33 2021

@author: oislen
"""

import os
import sys
scripts_dir = os.path.dirname(os.getcwd())
sys.path.append(scripts_dir)

import cons
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from utilities.perf_metrics import perf_metrics
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler

# load in data
base = pd.read_csv(cons.base_engin_data_fpath, 
                   sep = '|'
                   )

# split the data based on the original dataset
base_train = base[base.Dataset == 'train']
base_test = base[base.Dataset == 'test']

# extract out relevant columns
y_col = cons.y_col
X_col =  base.columns.drop(cons.id_cols).tolist()

# split the training data
X_train, X_valid, y_train, y_valid = train_test_split(base_train[X_col], 
                                                      base_train[y_col], 
                                                      test_size = 0.2, 
                                                      random_state = cons.random_state
                                                      )

# ranodom oversample target
ros = RandomOverSampler(random_state = cons.random_state)
X_train, y_train = ros.fit_resample(X_train, y_train)

# create test set
X_test = base_test[X_col]

# creat XD matrices for Xgboost model
XD_train = xgb.DMatrix(X_train, label = y_train['Survived'])
XD_valid = xgb.DMatrix(X_valid, label = y_valid['Survived'])
XD_test = xgb.DMatrix(X_test)

params = {'objective':'binary:hinge',
          'max_depth':13, 
          'learning_rate':0.1, 
          'eval_metric':'auc',
          'min_child_weight':7, 
          'subsample':0.8,
          'colsample_bytree':0.8,
          'seed':cons.random_state,
          'reg_alpha':0,
          'gamma':0, 
          'scale_pos_weight':1,
          'n_estimators': 5000,
          'nthread':-1
          }

# create watch list
watchlist = [(XD_train, 'train'), (XD_valid, 'valid')]

# define number of rounds
nrounds = 10000  

# fit xgboost model
model = xgb.train(params, 
                  XD_train, 
                  nrounds, 
                  watchlist, 
                  early_stopping_rounds = 50, 
                  maximize = True, 
                  verbose_eval = 10
                  )

y_valid['Predicted'] = model.predict(XD_valid)

val_metrics = perf_metrics(y_obs = y_valid['Survived'], 
                           y_pred = y_valid['Predicted'], 
                           target_type = 'class'
                           )

# create feature importance plot
fig, ax = plt.subplots(figsize=(15, 20))
xgb.plot_importance(model,ax=ax,max_num_features=20,height=0.8,color='g')
plt.show()

# visulaise the model
xgb.to_graphviz(model)

# make test set predictions
base_test['Survived'] = model.predict(XD_test).astype(int)

# output predictions
results = base_test[['PassengerId', 'Survived']]

# write predictions
results.to_csv(cons.pred_data_fpath.format('xgb'),
               sep = ',',
               encoding = 'utf-8',
               header = True,
               index = False
               )