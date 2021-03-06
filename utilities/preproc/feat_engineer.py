# -*- coding: utf-8 -*-
"""
Created on Sun Nov  4 21:46:26 2018

@author: oislen
"""

# load in relevant libraries
import os
import pandas as pd
import cons
from preproc.derive_variables import derive_variables
from preproc.tree_feat_imp import tree_feat_imp
from preproc.standardise_variables import standardise_variables

def feat_engineer(base_clean_fpath,
                  base_engin_fpath,
                  top_n_int_terms = 10,
                  stand_range = [0, 1],
                  stand_type = 'range'
                  ):
    
    """
    
    Feature Engineer Documenation
    
    Function Overview
    
    This function generates new features for the cleaned base data.
    
    Defaults
    
    feat_engineer(base_clean_fpath,
                  base_engin_fpath,
                  top_n_int_terms = 10,
                  stand_range = [0, 1],
                  stand_type = 'range'
                  )
    
    Parameters
    
    base_clean_fpath - String, the input file path to the cleaned base data
    base_engine_fpath - String, the output file path to write the engineered base data
    top_n_int_terms - Integer, the number of best interaction terms to return, default is 10
    stand_range - List of Integers, the standardisation range to standardise all predictor variables by, default is [0, 1]
    stand_type - String, the type of data standardisation to perform, default is 'range'
    
    Returns
    
    0 for successful execution
    
    Example
    
    feat_engineer(base_clean_fpath = 'C:\\Users\\...\\base_clean.csv',
                  base_engin_fpath = 'C:\\Users\\...\\base_engin.csv',
                  top_n_int_terms = 10,
                  stand_range = [0, 1],
                  stand_type = 'range'
                  )
    
    """
    
    print('checking inputs ...')
    
    # check input data types
    str_inputs = [base_clean_fpath, base_engin_fpath]
    if any([type(val) != str for val in str_inputs]):
        raise ValueError('Input params [base_clean_fpath, base_engin_fpath] must be str data types')
    # check if input file path exists
    if os.path.exists(base_clean_fpath) == False:
        raise OSError('Input file path {} does not exist'.format(base_clean_fpath))
    
    print('Loading base data ...')
    
    # load in data
    base = pd.read_csv(base_clean_fpath, 
                       sep = cons.sep
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
    
    # create the concatenation object list
    concat_objs = [base[base_cols], int_data]
    
    # create the engineered data by concatenating the base data with the interaction data
    engin = pd.concat(objs = concat_objs, axis = 1)

    print('Performing feature importance on all terms ...')
    
    # create a list of all predictors
    pred_cols = attr_cols + int_data.columns.tolist()
    
    # extract training data
    y_train = engin.loc[engin['Dataset'] == 'train', cons.y_col[0]]
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
    top_int_feat = feat_imp_sub['Predictor'].head(top_n_int_terms).tolist()
    
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
                                  stand_type = stand_type,
                                  stand_range = stand_range
                                  )
    
    # update the processed data
    final_data[stand_cols] = stand

    print('Outputting data ...')
    
    # output the dataset
    final_data.to_csv(base_engin_fpath,
                      sep = cons.sep,
                      encoding = cons.encoding,
                      header = cons.header,
                      index = cons.index
                      )
    
    return 0
