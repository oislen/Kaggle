# -*- coding: utf-8 -*-
"""
Created on Sun Nov  4 22:19:35 2018

@author: oislen
"""

# load in relevant libraries
import pandas as pd
from sklearn import ensemble
import cons
import value_analysis as va

def class_gm(base_engin_fpath,
             pred_fpath
             ):
    
    """
    
    Classification GBM Documentation
    
    Function Overview
    
    This function trains a GBM model classifying the test data
    
    Defaults
    
    class_gm(base_engin_fpath,
             pred_fpath
             )
    
    Parameters
    
    base_engin_fpath - String, the input file path for the base engineered data
    pred_fpath - String, the output file path for the GBM classifications
    
    Returns
    
    0 for successful execution
    
    Example
    
    class_gm(base_engin_fpath = 'C:\\Users\\...\\base_engin.csv',
             pred_fpath = 'C:\\Users\\...\\preds.csv'
             )
        
    """
    
    # load in data
    base = pd.read_csv(base_engin_fpath, 
                       sep = '|'
                       )
    
    # split the data based on the original dataset
    base_train = base[base.Dataset == 'train']
    base_test = base[base.Dataset == 'test']

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
    
    # tune gbm model
    mod_tuning = va.tune_hyperparameters(model = gbm, 
                                         params = cons.model_gbm_params, 
                                         X_train = X_train, 
                                         y_train = y_train,
                                         scoring = 'accuracy',
                                         verbose = 3
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
                                              n_estimators = best_params['n_estimators'],
                                              random_state = 123
                                              )
    
    # fit the model given best parameters for validation
    gbm.fit(X_train, 
            y_train.values.ravel()
            )
    
    # classify the validation set
    y_valid['Survived_pred'] = gbm.predict(X_valid)
    
    # genrate the regression metrics
    print(va.perf_metrics(y_obs = y_valid['Survived'], 
                          y_pred = y_valid['Survived_pred'], 
                          target_type = 'class'
                          ))
    
    # create a ROC curve
    va.Vis.roc_curve(obs = 'Survived', 
                     preds = 'Survived_pred', 
                     dataset = y_valid
                     )
  
    
    # randomly split the dataset
    X_valid, X_train, y_valid, y_train = va.train_test_split_sample(dataset = base_train,
                                                                    y = ['Survived'],
                                                                    X = base_train.columns.drop(['PassengerId', 'Dataset', 'Survived']),
                                                                    train_size = 1,
                                                                    test_size = 0,
                                                                    random_split = False,
                                                                    sample_target = 'Survived',
                                                                    sample_type = 'over'
                                                                    )
    
    # refit model to all training data
    gbm.fit(X_train, 
            y_train.values.ravel()
            )
    
    # predict for the test set
    base_test['Survived'] = gbm.predict(base_test[X_valid.columns])
    
    print('Outputting classifictions ...')
    
    # create the test classification dataset
    predictions = pd.DataFrame()
    predictions['Survived'] = base_test['Survived'] .astype(int)
    predictions['PassengerId'] = base_test.index + 1
    predictions = predictions[['PassengerId', 'Survived']]
    
    # output the dataset
    predictions.to_csv(pred_fpath,
                       sep = ',',
                       encoding = 'utf-8',
                       header = True,
                       index = False
                       )
    
    return 0

if __name__ == '__main__':
    
    class_gm(base_engin_fpath = cons.base_engin_data_fpath,
             pred_fpath = cons.pred_data_fpath
             )
