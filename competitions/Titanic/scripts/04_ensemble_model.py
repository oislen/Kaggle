# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 12:32:37 2021

@author: oislen
"""

# load relevant libraries
import cons
import pandas as pd
from sklearn import ensemble
from corr_mat import corr_mat
from ensemble.load_class_preds import load_class_preds
from ensemble.load_class_models import load_class_models
from ensemble.comp_valid_perf_metrics import comp_valid_perf_metrics

def ensemble_model():
    
    """
    
    Ensemble Model Documentation
    
    Function Overview
    
    This funcion fits an ensemble voting classifer given all the previous fitted sklearn single classifiers.
    
    Defaults
    
    ensemble_model()
    
    Parameters
    
    Returns
    
    0 for successful execution
    
    Example
    
    ensemble_model()
    
    
    """
    
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
    
    print('Loading model predictions ...')
    
    # load in the model predictions
    preds_df = load_class_preds(model_keys = model_keys, 
                                join_col = id_col,
                                pred_data_fpath = model_pred_data_fpath
                                )
    
    # check null values
    preds_df.isnull().sum()
    
    # extract out the prediction columns
    model_pred_cols = preds_df.columns.drop(id_col).tolist()
    
    print('Creating correlation matrix of model predictions ...')
    
    # plot correlation matrix of classification model predictions
    corr_mat(dataset = preds_df,
             attrs = model_pred_cols,
             method = 'spearman'
             )
        
    print('Creating comparison report of model performance metrics ...')
    
    # load performance metrics
    perf_metrics = comp_valid_perf_metrics(model_keys = sur_dict, 
                                           perf_metrics_fpath = cons.perf_metrics_fpath
                                           )
    
    print(perf_metrics)
    
    #-- Voting Classifier --#
    
    print('Creating majority vote classifer ...')
    
    # find majority vote
    preds_df['major_vote'] = preds_df.mode(numeric_only = True, axis = 1)
    
    # output predictions
    maj_vote = preds_df[[id_col, 'major_vote']].rename(columns = {'major_vote':'Survived'})
    
    # write predictions
    maj_vote.to_csv(model_pred_data_fpath.format('mvc'),
                    sep = cons.sep,
                    encoding = cons.encoding,
                    header = cons.header,
                    index = cons.index
                    )
    
    print('Loading engineered feature data ...')
    
    # load in the engineered data
    base_engin_fpath = cons.base_engin_data_fpath
      
    # load in data
    base = pd.read_csv(base_engin_fpath, 
                       sep = cons.sep
                       )
    
    print('Splitting into train and test sets ...')
    
    # split the data based on the original dataset
    base_train = base[base['Dataset'] == 'train']
    base_test = base[base['Dataset'] == 'test']
    
    print('Loading best sklearn models ...')
    
    # load in classification models
    class_models_dict = load_class_models(model_keys = model_keys, 
                                          model_fpath = model_class_fpath
                                          )
    
    print('Creating voting classifer model ...')
    
    # extract the estimaters from the model dictionary
    estimators = [(key, val) for key, val in class_models_dict.items()]
    
    # define the voting classifer model
    votingC = ensemble.VotingClassifier(estimators = estimators, 
                                        voting = 'soft', 
                                        n_jobs = -1
                                        )
    
    # definte the predictor columns
    X_col = base.columns.drop(cons.id_cols).tolist()
    
    print('Checking randomforest predictions as a sanity check ... ')
    
    # make test set predictions
    model = class_models_dict['rfc']
    preds_df['rfc2_Survived'] = model.predict(base_test[X_col]).astype(int)
    tab = pd.crosstab(index = preds_df['rfc_Survived'], columns = preds_df['rfc2_Survived'] )
    
    print(tab)
    
    print('Fitting voting classifier model ...')
    
    # fit voting classifer
    votingC = votingC.fit(X = base_train[X_col], 
                          y = base_train[cons.y_col[0]]
                          )
    
    print('predicting for test set ...')
    
    # make test set predictions
    preds_df['Survived'] = votingC.predict(base_test[X_col]).astype(int)
    
    # output predictions
    results = preds_df[[id_col, 'Survived']]
    
    # create a cross tab of differences between the majority vote and voting classifer model predicitons
    tab = pd.crosstab(index = preds_df['major_vote'], columns = preds_df['Survived'] )
    
    print(tab)
    
    print('Writing voting classifer model predictions to disk ...')
    
    # write predictions
    results.to_csv(model_pred_data_fpath.format('evc'),
                   sep = cons.sep,
                   encoding = cons.encoding,
                   header = cons.header,
                   index = cons.index
                   )
    
    return 0 

# if running main programme
if __name__ == '__main__':
    
    # run ensemble model
    ensemble_model()