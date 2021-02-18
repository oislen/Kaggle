# -*- coding: utf-8 -*-
"""
Created on Sat Nov  3 11:21:09 2018

@author: oislen
"""

#~~~~~ User Inputs ~~~~~

des_stats = True
vis_plots = True

#~~~~~~~~~~~~~~~~~~~~~~~

"""
#####################
#-- Preliminaries --#
#####################
"""

# load in relevant libraries
import pandas as pd
import numpy as np

# load cusotm functions
import sys
va_dir = 'C:/Users/User/Documents/Data_Analytics/Python/value_analysis'
sys.path.append(va_dir)
import value_analysis as va

# define the report directory
report_dir = 'C:\\Users\\User\\Documents\\GitHub\\Kaggle\\HousePrices\\report\\base_data\\'

# define the output path for the visualisations
output_vis_path = report_dir + 'plots\\'

# define the output path for the descript statistics
output_desc_stats_path = report_dir + 'descriptive_stats\\'

# load in data
input_dir = 'C:\\Users\\User\\Documents\\GitHub\\Kaggle\\HousePrices\\data\\'
base_name = 'base.csv'
base = pd.read_csv(input_dir + base_name, sep = '|')

# seperate out the different column types
cat_vars = base.dtypes[base.dtypes == 'object'].index.tolist()
int_vars = base.dtypes[base.dtypes == 'int64'].drop('Id').index.tolist()
float_vars = base.dtypes[base.dtypes == 'float64'].index.tolist()
num_cols =  float_vars + int_vars
str_cols =  base.dtypes.index[(base.dtypes == 'object')].tolist()

"""
################
#-- MetaData --#
################
"""

# dataset dimensions
base.shape

# dataset meta data
base.info()

# print the column names
[val for val in base.columns]

# deterine the data types of each column
base.dtypes


"""
#############################
#-- Decriptive Statistics --#
#############################
"""

if des_stats == True:
    
    ###############
    #-- Overall --#
    ###############
    
    # generate numeric statistics - overall
    numeric_stats = va.desc.stats(dataset = base, 
                                  attrs = base.columns, 
                                  stats_type = "numeric",
                                  output_dir = output_desc_stats_path,
                                  output_fname = "descriptive_statistics_numeric.csv"
                                  )
    
    # generate categorical statistics
    string_stats = va.desc.stats(dataset = base, 
                                 attrs = base.columns, 
                                 stats_type = "string",
                                 path = output_desc_stats_path
                                 )
    
    #################
    #-- Group Bys --#
    #################
    
    # define a list of variables to create group by statistics for
    groupby_vars = cat_vars + int_vars
    
    # for each variable in the groupby variable list
    for var in groupby_vars:
        
        # print an update
        print('~~~~~~ Working on ' + var)
        
        # create the output path
        output_dir = output_desc_stats_path + 'categorical_group_bys//'
        
        # create the output file name
        fname =  var.lower() + "_groupby_num_stats.csv"
        
        # group by neighbourhood
        numeric_stats = va.desc.stats(dataset = base, 
                                      attrs = ['SalePrice'], 
                                      stats_type = "numeric",
                                      groupby = [var],
                                      output_dir = output_dir,
                                      output_fname = fname
                                      )
        
    
    #########################
    #-- Association Tests --#
    #########################
    
    # extract out the string based columns
    col_str = base.dtypes[base.dtypes == 'object'].index
    
    # get the unique value counts for these columns
    n_str_lvls = base[col_str].apply(lambda x: x.nunique())
    
    # extract columns with less than 5 levels
    assoc_cols = n_str_lvls[n_str_lvls <= 5].index
    
    # cut up the sale price into categories
    train = base[base.Dataset == 'train']
    train_SalePrice = train['SalePrice']
    train['SalePriceCat'] = pd.qcut(train_SalePrice, q = 15).astype(str)
    
    # generate the association tests
    va.desc.var_assoc(dataset = base,
                      attrs = assoc_cols,
                      response = 'SalePriceCat',
                      assoc_type = 'nominal',
                      output_dir = report_dir + 'association_tests\\',
                      output_fname = 'nominal_association_tests_overall.csv'
                      )
    
    #########################
    #-- Correlation Tests --#
    #########################
    
    # generate the correlation tests
    va.desc.var_corr(dataset = base,
                     attrs = num_cols,
                     response = 'logSalePrice',
                     corr_type = 'spearman',
                     output_dir = report_dir + 'correlation_tests\\'
                     )

"""
######################
#-- Visualisations --#
######################
"""

if vis_plots == True:
    
    ##################
    #-- Histograms --#
    ##################
    
    va.visualise.hist(dataset = base,
                      num_var = num_cols,
                      bins = None,
                      hist = True,
                      kde = False,
                      output_dir = output_vis_path + '\\Histograms'
                      )
    
    ###################
    #-- Count Plots --#
    ###################
    
    va.visualise.count(dataset = base,
                       cat_var = str_cols,
                       output_dir = output_vis_path + '\\Countplots'
                       )
    
    
    ################
    #-- Boxplots --#
    ################
    
    va.visualise.boxplot(dataset = base,
                         num_var = ['logSalePrice'],
                         cat_var = str_cols,
                         output_dir = output_vis_path + '\\Boxplots'
                         )
    
    
    #####################
    #-- Scatter Plots --#
    #####################
    
    va.visualise.regplot(dataset = base, 
                         res_var = ['logSalePrice'],
                         pred_var = num_cols,
                         output_dir = output_vis_path + '\\Scatterplots'
                         )
