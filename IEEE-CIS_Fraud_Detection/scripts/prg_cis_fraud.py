# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 12:01:20 2019

@author: oislen
"""

#####################
#-- Preliminaries --#
#####################

# import relevant libraries
import pandas as pd
import sys

# import the static inputs
static_inputs = pd.read_csv('file:///C:/Users/User/Documents/GitHub/Kaggle/IEEE-CIS_Fraud_Detection/static_inputs.csv',
                            sep  = ',',
                            header = 0,
                            index_col = 0,
                            encoding = 'latin1'
                            )

# extract the project directory and the various sub directories
python_dir  = static_inputs.loc['python_dir', 'input']
value_analysis_subdir  = static_inputs.loc['value_analysis_subdir', 'input']

# create the various directories
value_analysis_dir =  python_dir + value_analysis_subdir

# import custom functions
sys.path.append(value_analysis_dir)
import value_analysis as va

#################
#-- Load Data --#
#################

# define a function to load in data
def load_data(data_type):
    
    """
    
    Program Load Raw data
    
    Function Overview
    
    This function loads in the raw competition data
    
    Parameters
    
    data_type - String, the type of data to import, either 'raw_data' and 'full_data'
    
    Returns
    
    if raw_data
        
        test_identity - DataFrame, the identity test data
        test_transaction - DataFrame, the transaction test data
        train_identity - DataFrame, the identity train data
        train_transaction - DataFrame, the transaction train data
    
    else if full_data
    
        full_data - DataFrame, the complete dataset inlcuding the train, test, identity and transaction
    
    Example
    
    prg_load_raw_data(data_type = 'raw_data)
    
    """
    
    # import the static inputs
    static_inputs = pd.read_csv('file:///C:/Users/User/Documents/GitHub/Kaggle/IEEE-CIS_Fraud_Detection/static_inputs.csv',
                                sep  = ',',
                                header = 0,
                                index_col = 0,
                                encoding = 'latin1'
                                )
    
    # extract the project directory and the various sub directories
    project_dir = static_inputs.loc['project_dir', 'input']
    data_subdir = static_inputs.loc['data_subdir', 'input']

    # create the various directories
    data_dir = project_dir + data_subdir

    # if loading in the raw data
    if data_type == 'raw_data':
        
        # print process update
        print('Loading test identity data ...')
        
        # load in the test identity data
        test_identity = pd.read_csv(data_dir + 'test_identity.csv')
        
        # print process update
        print('Loading test transaction data ...')
        
        # load in the test transaction data
        test_transaction = pd.read_csv(data_dir + 'test_transaction.csv')
        
        # print process update
        print('Loading train identity data ...')
        
        # load in the train identity data
        train_identity = pd.read_csv(data_dir + 'train_identity.csv')
        
        # print process update
        print('Loading train transaction data ...')
        
        # load in the train transaction data
        train_transaction = pd.read_csv(data_dir + 'train_transaction.csv')
        
        # return the various datasets
        return test_identity, test_transaction, train_identity, train_transaction
    
    # else if loading in the full dataset
    elif data_type == 'full_data':
        
        print('Loading full data ...')
        
        # load in the full dataset in chunks of 75k
        data_full = pd.read_csv(data_dir + 'data_all.csv',
                                sep = ',',
                                header = 0,
                                chunksize = 75000,
                                low_memory = False,
                                encoding = 'latin1'
                                )
        
        # concatenate the chunks together
        data_full = pd.concat(data_full, ignore_index = data_full)
        
        # return the full dataset
        return data_full
    
#########################
#-- Descriptive Stats --#
#########################

# define a function to create an inital EDA of the datasets
def desc_stats(dataset_dict, 
               output_report_dir
               ):
    
    """
    
    Initial EDA Documentation
    
    Function Overview
    
    This function generates a high level EDA report of all four datatsets.
    
    Parameters
    
    dataset_dict - Dictionary of output filenames and dataset, the output filename format and the corresponding datasets 
    stats_types_list - List of strings, the type of statistics to derive
    
    Return
    
    Outputs the reports
    
    Exmaple
    
    initial_EDA(dataset_dict, stats_types_list)
    
    """

    # create a list of stats_types
    stats_types_list = ['numeric', 'string']

    # create the descriptive stats pattern
    desc_name_pat = '_{}_stats.'
            
    # for each dataset in the dataset list
    for output_name, data in dataset_dict.items():
        
        # for each stats type in the stats types list
        for stats_type in stats_types_list:
            
            # split out the '.'
            output_name_splits = output_name.split('.')
            
            # create the new output name format
            output_name_new = output_name_splits[0] + desc_name_pat + output_name_splits[1]

            # create the output filename
            output_filename = output_name_new.format(stats_type)
            
            # print process update
            print('Working on {} ...'.format(output_filename))
            
            # derive descriptive statistics for the four datasets
            va.Desc.stats(dataset = data,
                          attrs = data.columns,
                          stats_type = stats_type,
                          digits = 3,
                          output_dir = output_report_dir,
                          output_fname = output_filename
                          )
            

#########################
#-- Corrleation Stats --#
#########################

# define a function to create an inital EDA of the datasets
def corr_stats(dataset_dict, 
               output_report_dir
               ):
    
    """
    
    Initial EDA Documentation
    
    Function Overview
    
    This function generates a variety of correlation statistics for a given dictionary of dataset names and datasets
    
    Parameters
    
    dataset_dict, 
    output_report_dir,
    response
    
    Return
    
    Exmaple
    
    """

    # create a list of stats_types
    corr_types_list = ['pearson', 'spearman', 'kendall']

    # create the descriptive stats pattern
    stats_name_pat = '_{}_stats.'
            
    # for each dataset in the dataset list
    for output_name, data in dataset_dict.items():
        
        # for each stats type in the stats types list
        for corr_type in corr_types_list:
            
            # split out the '.'
            output_name_splits = output_name.split('.')
            
            # create the new output name format
            output_name_new = output_name_splits[0] + stats_name_pat + output_name_splits[1]

            # create the output filename
            output_filename = output_name_new.format(corr_type)
            
            # print process update
            print('Working on {} ...'.format(output_filename))
            
            # derive descriptive statistics for the four datasets
            va.Desc.corr(attrs = data.columns,
                         dataset = data,
                         response = None,
                         corr_type = corr_type,
                         seed = 1234,
                         n_sample = 100000,
                         output_dir = output_report_dir,
                         output_fname = output_filename
                         )