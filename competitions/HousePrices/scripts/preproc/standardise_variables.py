# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 16:35:54 2021

@author: oislen
"""

import numpy as np

def standardise_variables(dataset,
                          attr = None,
                          stand_type = 'range',
                          stand_range = [0, 1],
                          scale = None,
                          power = None,
                          output_dir = None,
                          output_fname = None
                          ):
    
    """
    
    Standardise Variables Documentation
    
    Function Overview
    
    This function standardises the given variables of a dataset.
    Currenlty four types of standardisation are support; range, norm, mad, robust or rank.
    Range standardisation standardises variables to a specified range.
    Norm standardisation standardises variables to have mean 0 and standard deviattion 1.
    Median Absolute Deviation (MAD) standardises varables to have median 0 and median absolute divation of 1.
    Robust standardisation standardises removes the median and normalises by the interquartile range.
    Rank standardisation ranks the values from 0 onwards and divides by the maximum rank.
    If no attributes are given, the entire dataset is standardised.
    
    Defaults
    
    standardise_variables(dataset,
                          attr = None,
                          stand_type = 'range',
                          stand_range = [0, 1],
                          output_dir = None,
                          output_fname = None
                          )
    
    Parameters
    
    dataset - Dataframe, the dataset
    attr - List of Strings, the attributes from the dataset to standardise
    stand_type - String, the type of standardisation, either 'range', 'norm', 'mad', 'robust' or 'rank'.
    stand_range - List of two floats, the range to standardise to
    output_dir - String, the directory to output the file to, default is None.
    output_fname - String, the filename to output the file as, must include a file suffix, default is None.
    
    Output
    
    pandas.DataFrame - Dataset with the standardised variables
    
    Example
    
    va.standardise_variables(dataset = Simply_Business, 
                             stand_type = 'mad',
                             attr = ['lr_f_score_pcd']
                             )
    
    See Also
    
    va.derive_variables
    
    Reference
    
    https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html
    https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
    https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html

    """
    
    #####################
    #-- Preliminaries --#
    #####################
    
    # import the relevant libraries
    #import pandas as pd

    ######################
    #-- Error Handling --#
    ######################
    
    # error handling for 'stand_type'
    if stand_type not in ['range', 'norm', 'mad', 'robust', 'rank', 'scale', 'score']:
        
        # print error message and end script
        message = "Error: incorrect stand_type given; " + stand_type + ". Must be either 'range', 'mean', 'robost', 'mad', 'rank', 'scale' or 'score'"
        
        raise ValueError(message)
    
    # use a for loop to loop through the attribute values
    for val in attr:
    
        # check the attribute column is in the dataset
        if val not in dataset.columns:
            
            # print error message and end script
            message = "Error: The specified attribute column " + val + " is not a column from the dataset"
            
            raise ValueError(message)
    
    #######################
    #-- Standardisaiton --#
    #######################
    
    # if no attributes are give
    if (attr == None):
        
        # use the whole dataset
        attr = dataset.columns.tolist()

    # extract out the numeric datasets
    int_cols = dataset[attr].columns[dataset[attr].dtypes == 'int64'].tolist()
    float_cols = dataset[attr].columns[dataset[attr].dtypes == 'float64'].tolist()
    num_cols = int_cols + float_cols

    # extract out the pandas series
    series = dataset[num_cols]
        
    # if the standardisation type is range
    if (stand_type == 'range'):
        
        # create the upper and lower bounds
        upper_bound = max(stand_range)
        lower_bound = min(stand_range)

        # perform range standardisation
        numerator = (series - series.min()) * (upper_bound - lower_bound)
        denominator = series.max() - series.min()
        denominator[denominator == 0 ] = 1
        stand_df =  (numerator / denominator) + lower_bound
    
    # if the standardisation type is norm
    elif (stand_type == 'norm'):
        
        # perform norm standardisation
        numerator = series - series.mean()
        denominator = series.std()
        denominator[denominator == 0 ] = 1
        stand_df = numerator / denominator
    
    # if the standardisation type is mad
    elif (stand_type == 'mad'):
        
        # perform mad standardisation
        numerator = series - series.median()
        denominator = (series.abs() - series.median()).median()
        denominator[denominator == 0 ] = 1
        stand_df = numerator / denominator
        
    # if the standardise function type is robust
    elif (stand_type == 'robust'):
        
        # perform the robust standardisation
        numerator = series - series.median()
        denominator = series.quantile(q = 0.75) - series.quantile(q = 0.25)
        denominator[denominator == 0 ] = 1
        stand_df = numerator / denominator   
    
    # else if the standardisiation funciton is index_rank
    elif (stand_type == 'rank'):
        
        # perform the index rank standardisation
        numerator = series.rank(method = 'min') - 1
        denominator = numerator.max()
        stand_df = numerator / denominator  
        # else if the standardisiation funciton is index_rank
        
    elif (stand_type == 'scale'):
        
        # perform the scale standardisation
        numerator = series
        denominator = series.max() - series.min()
        stand_df = numerator / denominator  
    
    # else if standardising to score specs
    elif (stand_type == 'score'):
        
        # create the upper and lower bounds
        score_max = max(stand_range)
        score_min = min(stand_range)
        
        # lifted from Fionn & Xumue's work on Spain
        # create score
        scalar = (score_max - score_min) 
        exp = np.exp(-(series / scale) * power)
        stand_df = score_min + scalar * exp
    
    ##############
    #-- Output --#
    ##############
    
    # if output directory is given
    if (output_dir != None):
        
        # if output filename is not given
        if output_fname == None:
            
            # assign the default filename
            fname = 'standardised_variables_' + stand_type + '.csv'
            
        # else if the output filename is given
        elif output_fname != None:
            
            # assign the given filename
            fname = output_fname
        
        # create the full output path
        output_path = output_dir + '/' + fname
        
        # output the file using the path
        stand_df.to_csv(output_path,
                        sep = '|',
                        index = False,
                        header = True,
                        encoding = 'latin'
                        )
    
    # return the derive_df
    return stand_df
