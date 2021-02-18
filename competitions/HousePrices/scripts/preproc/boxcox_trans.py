# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 17:38:12 2021

@author: oislen
"""

#####################
#-- Preliminaries --#
#####################
    
# import relevant libraries
import pandas as pd
import scipy.stats as stats
import value_analysis.constants as cons
import matplotlib.pyplot as plt

plot_size_width = cons.plot_size_width
plot_size_height = cons.plot_size_height
plot_size = (plot_size_width, plot_size_height)
title_size = cons.plot_title_size
axis_text_size = cons.plot_axis_text_size
labelsize = cons.plot_label_size

###########################
#-- Function Definition --#
###########################

def boxcox_trans(dataset,
                 attr,
                 probplot = True,
                 output_dir = None,
                 output_fname = None
                 ):
    
    """
    
    Box-Cox Power Transformation Documentation
    
    Function Overview

    This function performs a box cox power transformation on a given set of attributes.
    The box-cox power transformation is a normality power transformation technique for numeric attributes.
    The transformed variables are returned as a pandas dataframe.
    Optionally, prior and subsequent normal Q-Norm plots 
     
    Defaults
    
    boxcox_trans(dataset,
                 attr,
                 probplot = True,
                 output_dir = None,
                 output_fname = None
                 )
    
    Parameters
    
    dataset - Dataframe, the dataset to derive attributes from.
    attr - List of strings, the names of the variables to derive variables from.
    probplot - Boolean, whether to plot a prior and subsequent probability plots for the transformed attributes, default is True.
    output_dir - String, the output directory to save the derived data as a .csv file, default is None
    output_fname - String, the output file name for the derived data to be saved as a .csv file, default is None

    Returns
    
    A dataframe with the boxcox transformed variables.
    
    Example
    
    boxcox_trans(dataset = SimplyBusiness,
                 attr = ['sil_wealth_score_pc, hr_mths_last_addr_any, hr_n_unique_uklexids_eer]
                 )
    
    References
    
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.boxcox.html
    
    """
    
    ######################
    #-- Error Handling --#
    ######################
    
    # extract the column names from the dataset
    data_columns = dataset.columns
    
    # check the attribute column is in the dataset
    missing_cols_bool = [val not in data_columns for val in attr]
        
    # if the column is not in the dataset
    if any(missing_cols_bool):
        
        # extract out the missing columns
        missing_cols = [val for val in attr if val not in data_columns]
        
        # raise a value error
        raise ValueError("Input Error: The specified attribute column(s) {} is not a column from the dataset.".format(str(missing_cols)))

    # check the data types of the attributes
    object_dtypes = [dataset[val].dtypes == 'object' for val in attr]
    
    # if any of the attributes are object data types
    if any(object_dtypes):
        
        # extract out the object data type columns
        object_cols = [val for val in attr if dataset[val].dtypes == 'object']
        
        # raise a value error
        raise ValueError("Input Error: The specified attribute column(s) {} is an object data types.".format(str(object_cols)))
        
    ###########################
    #-- Variable Derivation --#
    ###########################
    
    print('~~~~~ Transforming variables ...')
    
    # create a list of the numeric columns
    int_col_bool = (dataset.dtypes[attr] == 'int64')
    float_cols_bool = (dataset.dtypes[attr] == 'float64')      
    dev_cols = dataset[attr].columns[int_col_bool | float_cols_bool].tolist()
        
    # create an empty dataframe to hold the derived data
    derive_df = pd.DataFrame()
    
    # for each column in the numeric columns list
    for col in dev_cols:
        
        # print process update
        print(col)
        
        # extract the series from the datatframe
        series = dataset[col]
        
        # calculate the optimal lambda for the box cox transformation
        lam = stats.boxcox_normmax(x = series + 1,
                                   method = 'mle'
                                   )
        
        # transform the attribute using the optimal lambda
        boxcox_series = stats.boxcox(series + 1, lam)
        
        # assign the box-cox transformed series to the derived dataframe
        derive_df[col] = boxcox_series
        
        # if creating a probability plots
        if probplot == True:
            
            # define the font dictionary
            fontdict = {'fontsize':axis_text_size}
            
            # create the probability plot
            fig = plt.figure(figsize = plot_size)
            ax = fig.add_subplot(111)
            stats.probplot(x = series, dist = stats.norm, plot = ax)
            ax.set_title(label = 'Prior Normal Probablity Plot for {}'.format(col), fontdict = fontdict)
            ax.set_ylabel(ylabel = 'Ordered Values', fontdict = fontdict)
            ax.set_xlabel(xlabel = 'Theoretical Quantiles', fontdict = fontdict)
            plt.show()
        
            # create the probability plot
            fig = plt.figure(figsize = plot_size)
            ax = fig.add_subplot(111)
            stats.probplot(x = boxcox_series, dist = stats.norm, plot = ax)
            ax.set_title(label = 'Subsequent Normal Probablity Plot for {}'.format(col), fontdict = fontdict)
            ax.set_ylabel(ylabel = 'Ordered Values', fontdict = fontdict)
            ax.set_xlabel(xlabel = 'Theoretical Quantiles', fontdict = fontdict)
            plt.show()

    ##############
    #-- Output --#
    ##############
                
    print('~~~~~ Outputting dataframe ...')
                
    # option: save the plot
    if (output_dir != None):
        
        # if the ouptut file name is not given
        if (output_fname == None):
            
            # define the output filename and path 
            output_filename = 'boxcox_transformed_var.csv'
                
        # else if the output filename is given
        elif (output_fname != None):
            
            # define the output filename and path 
            output_filename = output_fname
        
        # create the output path
        output_path = output_dir + output_filename
        
        # save the data frame as latin encoded "|" seperated .csv file
        derive_df.to_csv(output_path,
                         sep = '|',
                         encoding = 'utf-8',
                         header = True,
                         index = True
                         )
    
    # return the derive_df
    return(derive_df)
