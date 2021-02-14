# -*- coding: utf-8 -*-
"""
Created on Sun Nov  4 21:46:26 2018

@author: oislen
"""

# load in relevant libraries
import pandas as pd
from sklearn import ensemble
import cons
from utilities.derive_variables import derive_variables
from utilities.tree_feat_imp import tree_feat_imp
from utilities.standardise_variables import standardise_variables

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
    
    # extract out the base columns
    base_cols = base.columns.tolist()
    
    # extract out the integer columns
    attr_cols = [col for col in base_cols if col not in cons.id_cols]
    
    # create interaction terms
    int_data = derive_variables(dataset = base,
                                attr = attr_cols,
                                var_type = 'interaction'
                                )
    
    # create the engineered data by concatenating the base data with the interaction data
    engin = pd.concat(objs = [base[base_cols], int_data],
                      axis = 1
                      )

    print('Performing feature importance on all terms ...')
    
    # create a list of all predictors
    pred_cols = attr_cols + int_data.columns.tolist()
    
    # extract training data
    y_train = engin.loc[engin['Dataset'] == 'train', 'Survived']
    X_train = engin.loc[engin['Dataset'] == 'train', pred_cols]
    
    # create a tree model
    model = cons.sur_rfc_mod
    
    # determine the feature importance
    feat_imp = tree_feat_imp(model = model,
                             y_train = y_train,
                             X_train = X_train
                             )
    
    # consider only interaction terms
    int_feat_imp_filt = pd.Series(feat_imp.index).str.contains('_x_').tolist()
    
    # filter out non interaction terms from feat_imp
    feat_imp_sub = feat_imp.loc[int_feat_imp_filt, :].reset_index()
    
    # extract out the important features 
    top_int_feat = feat_imp_sub['Predictor'].head(10).tolist()
    
    # add in additional variables to enable the interaction effects
    out_vars = base_cols + top_int_feat
    
    # create the final dataset
    final_data = engin[out_vars]

    print('Standardising data ...')
    
    # define the columns to standardise
    stand_cols = attr_cols + top_int_feat
    
    # standardise data to interval [0, 1]
    stand = standardise_variables(dataset = final_data,
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
