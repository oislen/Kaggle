# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 12:32:37 2021

@author: oislen
"""

# load relevant libraries
import cons
import pandas as pd
from sklearn import ensemble
from graph.corr_mat import corr_mat
from ensemble.load_class_preds import load_class_preds
from ensemble.load_class_models import load_class_models

# extract out the surivial model information
sur_dict = cons.sur_dict.keys()

# define a list of models to ignore
ignore_models = ['dtc']

# create a list of model keys
model_keys = [model for model in sur_dict if model not in ignore_models]

# define the joining column for classification predictions
id_col = 'PassengerId'

# create the full input file path to the model predictions
model_pred_data_fpath = cons.pred_data_fpath

# create the full input file path to the model objects
model_class_fpath = cons.best_model_fpath

#-- Correlation of Predictions --#

# load in the model predictions
preds_df = load_class_preds(model_keys = model_keys, 
                            join_col = id_col,
                            pred_data_fpath = model_pred_data_fpath
                            )

# check null values
preds_df.isnull().sum()

# extract out the prediction columns
model_pred_cols = preds_df.columns.drop(id_col).tolist()

# plot correlation matrix of classification model predictions
corr_mat(dataset = preds_df,
         attrs = model_pred_cols,
         method = 'spearman'
         )

# find majority vote
preds_df['major_vote'] = preds_df.mode(numeric_only = True, axis = 1)


# output predictions
maj_vote = preds_df[[id_col, 'major_vote']].rename(columns = {'major_vote':'Survived'})

# write predictions
maj_vote.to_csv(model_pred_data_fpath.format('mvc'),
                sep = ',',
                encoding = 'utf-8',
                header = True,
                index = False
                )

#-- Voting Classifier --#

# load in the engineered data
base_engin_fpath = cons.base_engin_data_fpath
  
# load in data
base = pd.read_csv(base_engin_fpath, 
                   sep = '|'
                   )

# split the data based on the original dataset
base_train = base[base.Dataset == 'train']
base_test = base[base.Dataset == 'test']

# load in classification models
class_models_dict = load_class_models(model_keys = model_keys, 
                                      model_fpath = model_class_fpath
                                      )

# extract the estimaters from the model dictionary
estimators = [(key, val) for key, val in class_models_dict.items()]

# define the voting classifer model
votingC = ensemble.VotingClassifier(estimators=estimators, 
                                    voting = 'soft', 
                                    n_jobs = 4
                                    )

# definte the predictor columns
X_col = base.columns.drop(cons.id_cols).tolist()

# fit voting classifer
votingC = votingC.fit(X = base_train[X_col], 
                      y = base_train[cons.y_col[0]]
                      )

# make test set predictions
preds_df['Survived'] = votingC.predict(base_test[X_col]).astype(int)

# make test set predictions
model = class_models_dict['rfc']
#t = model.estimators_[0].tree_
#t.impurity

#print(model.get_params())
#base_train[X_col].tail(10).to_csv('train2.csv')
#base_test[X_col].tail(10).to_csv('test2.csv')
#model.fit(X = base_train[X_col],  y = base_train[cons.y_col])

preds_df['rfc2_Survived'] = model.predict(base_test[X_col]).astype(int)

#preds_df[['rfc_Survived', 'rfc2_Survived']].tail(10)
#filt = preds_df['rfc_Survived'] != preds_df['rfc2_Survived']
#errors = preds_df.loc[filt, [id_col, 'rfc_Survived', 'rfc2_Survived']]
#error_pass = errors['PassengerId'].head(1)

#model.decision_path(base_test.loc[base_test['PassengerId'].isin(error_pass), X_col])


# output predictions
results = preds_df[[id_col, 'Survived']]

print(pd.crosstab(index = preds_df['major_vote'], columns = preds_df['Survived'] ))

# write predictions
results.to_csv(model_pred_data_fpath.format('evc'),
               sep = ',',
               encoding = 'utf-8',
               header = True,
               index = False
               )
