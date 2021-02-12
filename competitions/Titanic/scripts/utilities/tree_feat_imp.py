# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 16:35:08 2021

@author: oislen
"""

#####################
#-- Preliminaries --#
#####################

# import the relevant libraries
import pandas as pd
import numpy as np
from collections import OrderedDict

###############################
#-- Tree Feature Importance --#
###############################

def tree_feat_imp(X_train,
                  y_train, 
                  model,
                  output_dir = None,
                  output_fname = None
                  ):
    
    """
    
    Simple TREE Analysis Documentation

    Function Overview
    
    This function takes in a dataset and calculates the most import variables given a response variable.
    The function utilises various models from the ensemble module from scikit-learn, notably random forest, gradient boosting, ada boost and extra trees. 
    The response variable can be either a category or a numeric variable.
    Note, all predictor variables must be numeric and the dataset must be complete and have no missing data.
    
    Defaults
    
    TREE_feat_imp(X_train,
                  y_train, 
                  model,
                  output_dir = None,
                  output_fname = None
                  )
    
    Parameters
    
    X_train - Dataframe, the predictor data to calculate importance for
    y_train - Dataframe, the response data to calculate importance for
    model - Sklearn model, the classification or regression model to use
    output_dir - String, the output directory of the feature importance, default None.
    output_fname - String, the filename and extension of the output file, default is None.
    
    Returns
    
    A dataframe with the important variables
      
    Example
    
    TREE_feat_imp(y_train = (lungcap['Gender'] == 'male').astype(int),
                  X_train = lungcap[['Age', 'LungCap']], 
                  model = gbm,
                  output_dir = '/opt/dataprojects/Analysis',
                  output_fname = 'GAM_feat_imp_train.csv
                  )
    
    See Also
    
    GAM_feat_imp, GLM.simple_analysis, MARS_feat_imp, RGLM_feat_imp
    
    Reference
    
    https://scikit-learn.org/stable/modules/tree.html
    https://scikit-learn.org/stable/modules/ensemble.html#ensemble
    
    """
    
    # fit the random forest model
    model.fit(X_train, y_train)
        
    # calculate the importance of the predictor variables
    pred_import = np.round(model.feature_importances_,
                           decimals = 3) * 100
        
    # create a dataframe
    df_import = pd.DataFrame(OrderedDict({'Predictor':X_train.columns.tolist(),
                                          'Importance':pred_import}))
    
    # sort the the dataframe based on variable importance
    df_import = df_import.sort_values(by = 'Importance', 
                                      axis = 0, 
                                      ascending = False)
    
    # set the index
    df_import = df_import.set_index(keys = ['Predictor'])
    
    # if the output directory is given
    if output_dir != None:
        
        # if the output filename is given
        if output_fname != None:
            
            # set the filename
            filename = output_fname
            
        # else if the output filename is not given
        elif output_fname == None:
            
            # create the filename
            filename = 'TREE_feat_imp.csv'
        
        # create the absolute file path
        abs_file_path = output_dir + '/' + filename
        
        # save the feature importance
        df_import.to_csv(path_or_buf = abs_file_path,
                         sep = '|',
                         header = True,
                         index = True,
                         encoding = 'latin1'
                         )
        
    # return the importance dataframe
    return(df_import)
    
