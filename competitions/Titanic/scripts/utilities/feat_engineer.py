# -*- coding: utf-8 -*-
"""
Created on Sun Nov  4 21:46:26 2018

@author: oislen
"""

# load in relevant libraries
import pandas as pd
from sklearn import ensemble
import value_analysis as va

def feat_engineer(base_clean_2_fpath,
                  base_engin_fpath
                  ):
    
    """
    
    Feature Engineer Documenation
    
    Function Overview
    
    This function generates new features for the cleaned base data.
    
    Defaults
    
    feat_engineer(base_clean_2_fpath,
                  base_engin_fpath
                  )
    
    Parameters
    
    base_clean_2_fpath - String, the input file path to the cleaned base data
    base_engine_fpath - String, the output file path to write the engineered base data
    
    Returns
    
    0 for successful execution
    
    Example
    
    feat_engineer(base_clean_2_fpath = 'C:\\Users\\...\\base_clean_2.csv',
                  base_engin_fpath = 'C:\\Users\\...\\base_engin.csv'
                  )
    
    """
    
    # load in data
    base = pd.read_csv(base_clean_2_fpath, 
                       sep = '|'
                       )

    print('Deriving interaction terms ...')
    
    # define id columns
    id_cols = ['PassengerId', 'Dataset', 'Survived']
    
    # extract the columns to create interaction terms for
    int_cols = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'FamSize',
                'Alone', 'Mr', 'Mrs', 'Ms', 'Priv', 'male'
                ]
    
    # create interaction terms
    int_data = va.derive_variables(dataset = base,
                                   attr = int_cols,
                                   var_type = 'interaction'
                                   )
    
    # create the engineered data by concatenating the base data with the interaction data
    sub_cols = id_cols + int_cols
    engin = pd.concat(objs = [base[sub_cols], int_data],
                      axis = 1
                      )

    print('Performing feature importance on all terms ...')
    
    # randomly split the dataset
    X_valid, X_train, y_valid, y_train = va.train_test_split_sample(dataset = engin,
                                                                    y = ['Survived'],
                                                                    X = engin.columns.drop(['PassengerId', 'Dataset', 'Survived']),
                                                                    train_size = 0.8,
                                                                    test_size = 0.2,
                                                                    random_split = True,
                                                                    sample_target = 'Survived',
                                                                    sample_type = 'over'
                                                                    )
    
    # create a gbm model
    gbm = ensemble.GradientBoostingClassifier(random_state = 123)
    
    # determine the feature importance
    feat_imp = va.tree_feat_imp(model = gbm,
                                y_train = y_train,
                                X_train = X_train
                                )
    
    # consider only interaction terms
    int_feat_imp_filt = pd.Series(feat_imp.index).str.contains('_x_').tolist()
    
    # filter out non interaction terms from feat_imp
    feat_imp_sub = feat_imp.loc[int_feat_imp_filt, :]
    
    # extract out the important features 
    top_int_feat = feat_imp_sub.index[feat_imp_sub['Importance'] > 5].tolist()
    
    # add in additional variables to enable the interaction effects
    out_vars = id_cols + int_cols + top_int_feat
    
    # create the final dataset
    final_data = engin[out_vars]

    print('Standardising data ...')
    
    # define the columns to standardise
    stand_cols = int_cols + top_int_feat
    
    # standardise data to interval [0, 1]
    stand = va.standardise_variables(dataset = final_data,
                                     attr = stand_cols,
                                     stand_type = 'range',
                                     stand_range = [0, 1]
                                     )
    
    # update the processed data
    final_data[stand_cols] = stand

    print('Outputting data ...')
    
    # output the dataset
    final_data.to_csv(base_engin_fpath,
                      sep = '|',
                      encoding = 'utf-8',
                      header = True,
                      index = False
                      )
    
    return 0
