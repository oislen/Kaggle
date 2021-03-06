# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 16:33:36 2021

@author: oislen
"""

#-- Preliminaries --#

# import relevant libraries
import pandas as pd
import numpy as np

########################
#-- Derive variables --#
########################

def derive_variables(dataset,
                     attr,
                     var_type ='dummy',
                     power = 2,
                     suffix = None,
                     path = None
                     ):
    
    """
    
    Derive Variables Documentation
    
    Function Overview

    This function derives new attributes from existing ones from a given dataset.
    The function can derive either dummy, interaction or polynomial attributes.
     
    Defaults
    
    derive_variables(dataset,
                     attr,
                     var_type ='dummy',
                     power = 2,
                     suffix = '_bin',
                     path = None
                     )
    
    Parameters
    
    dataset - Dataframe, the dataset to derive attributes from.
    attr - List of strings, the names of the variables to derive variables from.
    var_type - String, the type of variable to be derived, either 'dummy', 'interaction', or 'polynomial', default is 'dummy'.
    power - Float, the polynomial degree to raise the specified attributes to, default is 2. 
    suffix - String, a suffix to add to the end of newly derived column names, default is None.
    path - String, the path to save the derived data as a .csv file, default is None

    Returns
    
    A dataframe with the derived variables
    
    Example
    
    derive_variables(dataset = SimplyBusiness,
                     attr = ['sil_wealth_score_pc, hr_mths_last_addr_any, hr_n_unique_uklexids_eer],
                     var_type = 'interaction'
                     )
    
    """

    ######################
    #-- Error Handling --#
    ######################
    
    # error handling for 'var_type'
    if var_type not in ['dummy', 'interaction', 'polynomial']:
        
        # print error message and end script
        raise ValueError("Error: incorrect var_type given, must be either 'dummy', 'interaction' or 'polynomial'")
    
    # check the attribute column is in the dataset
    for val in attr:
    
        if val not in dataset.columns:
            
            raise ValueError("Error: The specified attribute column " + val + " is not a column from the dataset")
    
    ###########################
    #-- Variable Derivation --#
    ###########################
    
    # if derving dummy variables
    if (var_type == 'dummy'):

        # filter out the str attributes
        dev_cols = dataset[attr].columns[dataset.dtypes[attr] == 'object']
        
    # else if deriving numeric varibles
    elif (var_type in ['polynomial', 'boxcox', 'interaction']):
        
        # create a list of the numeric columns
        int_col_bool = (dataset.dtypes[attr] == 'int64')
        float_cols_bool = (dataset.dtypes[attr] == 'float64')      
        dev_cols = dataset[attr].columns[int_col_bool | float_cols_bool].tolist()
        
    # if var_type is dummy
    if (var_type == 'dummy'):
                
        # derive the dummy variables
        derive_df = pd.get_dummies(data = dataset[dev_cols], 
                                   dummy_na = True,
                                   drop_first = True,
                                   dtype = np.int64
                                   )

    # if var_type is polynomial
    elif (var_type == 'polynomial'):

        # derive the polynomial attributes
        derive_df = dataset[dev_cols] ** power
            
    # if var_type is interaction
    elif (var_type == 'interaction'):
    
        # create an empty dataframe to hold the derived data
        derive_df = pd.DataFrame()
        
        # use a for loop to loop through the predictors
        for i, val in enumerate(dev_cols):
                    
            # create a j index
            j = i + 1
                
            # use a for loop to loop through the remaining predictors
            for pred in attr[j:]:

                # derive the interaction between the two variables
                derive_df[val + '_x_' + pred] = dataset[val] * dataset[pred] 
        
    # if a naming suffix is given
    if suffix != None:
        
        # extract the current columns
        col_names = derive_df.columns.tolist()
        
        # create the column names
        new_cols = [col + suffix for col in col_names]
        
        # create the renamining dictionary
        derive_df.columns = new_cols
        
    ##############
    #-- Output --#
    ##############
                
    # option: save the data
    if (path != None):
        derive_df.to_csv(path + 'derived_variables_' + var_type + '.csv')
    
    # return the derive_df
    return(derive_df)
