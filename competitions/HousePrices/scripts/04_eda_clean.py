# -*- coding: utf-8 -*-
"""
Created on Sat Nov  3 11:21:09 2018

@author: oislen
"""

#~~~~~ User Inputs ~~~~~

des_stats = True
feat_imp = True
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
import statsmodels.api as sm

# load cusotm functions
import sys
va_dir = 'C:/Users/User/Documents/Data_Analytics/Python/value_analysis'
sys.path.append(va_dir)
import value_analysis as va

# define the report directory
report_dir = 'C:\\Users\\User\\Documents\\GitHub\\Kaggle\\HousePrices\\report\\clean_data\\'

# define the output path for the visualisations
output_vis_path = report_dir + 'plots\\'

# define the output path for the descript statistics
output_desc_stats_path = report_dir + 'descriptive_stats\\'

# load in data
input_dir = 'C:\\Users\\User\\Documents\\GitHub\\Kaggle\\HousePrices\\data\\'
clean_name = 'clean.csv'
clean = pd.read_csv(input_dir + clean_name, sep = '|')

# seperate out the different column types
cat_vars = clean.dtypes[clean.dtypes == 'object'].index.tolist()
int_vars = clean.dtypes[clean.dtypes == 'int64'].index.tolist()
float_vars = clean.dtypes[clean.dtypes == 'float64'].index.tolist()
num_cols =  float_vars + int_vars
str_cols =  clean.dtypes.index[(clean.dtypes == 'object')].tolist()

"""
################
#-- MetaData --#
################
"""

# dataset dimensions
clean.shape

# dataset meta data
clean.info()

# print the column names
[val for val in clean.columns]

# deterine the data types of each column
clean.dtypes

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
    numeric_stats = va.desc.stats(dataset = clean, 
                                  attrs = clean.columns, 
                                  stats_type = "numeric",
                                  output_dir = output_desc_stats_path,
                                  output_fname = "descriptive_statistics_numeric.csv"
                                  )
    
    # generate categorical statistics
    string_stats = va.desc.stats(dataset = clean, 
                                 attrs = clean.columns, 
                                 stats_type = "string",
                                 output_dir = output_desc_stats_path
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
        numeric_stats = va.desc.stats(dataset = clean, 
                                      attrs = ['logSalePrice'], 
                                      stats_type = "numeric",
                                      groupby = [var],
                                      output_dir = output_dir,
                                      output_fname = fname
                                      )
        
    
    #########################
    #-- Association Tests --#
    #########################
    
    # extract out the string based columns
    col_str = clean.dtypes[clean.dtypes == 'object'].index
    
    # get the unique value counts for these columns
    n_str_lvls = clean[col_str].apply(lambda x: x.nunique())
    
    # extract columns with less than 5 levels
    assoc_cols = n_str_lvls[n_str_lvls <= 5].index
    
    # cut up the sale price into categories
    clean['SalePriceCat'] = pd.cut(clean.logSalePrice, bins = 5).astype(str)
    
    # generate the association tests
    va.desc.var_assoc(dataset = clean,
                      attrs = assoc_cols,
                      response = 'SalePriceCat',
                      assoc_type = 'nominal',
                      output_dir = report_dir + 'association_tests\\',
                      output_fname = 'nominal_association_tests_overall.csv'
                      )
    
    #########################
    #-- Correlation Tests --#
    #########################
    
    #-- Against Target --#
    
    # generate the correlation tests
    va.desc.var_corr(dataset = clean,
                     attrs = num_cols,
                     response = 'logSalePrice',
                     corr_type = 'spearman',
                     output_dir = report_dir + 'correlation_tests\\'
                     )


"""
##########################
#-- Feature Importance --#
##########################
"""

if feat_imp == True:
    
    print('Performing feature importance ...')
    
    # set the output directory
    feat_imp_dir = report_dir + 'feat_imp\\'
    
    # split the datasets
    test = clean[(clean['Dataset'] == 'test')]
    train = clean[(clean['Dataset'] == 'train')]
    
    # randomly split the dataset
    X_valid, X_train, y_valid, y_train = va.train_test_split_sample(dataset = train,
                                                                    y = ['logSalePrice'],
                                                                    X = train.columns.drop(['Id', 'logSalePrice', 'Dataset']),
                                                                    train_size = 0.8,
                                                                    test_size = 0.2
                                                                    )
    
    # calculate feature importance with GLM
    GLM_feat_imp = va.GLM.simple_analysis(endog = y_train,
                                          exog = X_train,
                                          family = sm.families.Gaussian(),
                                          output_dir = feat_imp_dir,
                                          output_fname = 'GLM_feat_imp.csv'
                                          )
    
    GLM_feat_imp
    
    # calculate feature importance with RGLM
    RGLM_feat_imp = va.RGLM_feat_imp(endog = y_train,
                                     exog = X_train,
                                     family = sm.families.Gaussian(),
                                     output_dir = feat_imp_dir,
                                     output_fname = 'RGLM_feat_imp.csv'
                                     )
    
    RGLM_feat_imp
    
    

"""
######################
#-- Visualisations --#
######################
"""

vis_plots == True:
    
    # extract the ordinal and binary variables
    ord_bin_cols = clean.columns[clean.columns.str.contains('_ord|_bin$')].tolist()
    
    # convert these to category variables
    clean[ord_bin_cols] = clean[ord_bin_cols].astype(str)
    
    # drop the ordinal and binary variables from the numeric variables list
    num_cols = [col for col in num_cols if col not in ord_bin_cols]
    
    ##################
    #-- Histograms --#
    ##################
    
    va.visualise.hist(dataset = clean,
                      num_var = num_cols,
                      bins = None,
                      hist = True,
                      kde = False,
                      output_dir = output_vis_path + '\\Histograms'
                      )
    
    ###################
    #-- Count Plots --#
    ###################
    
    va.visualise.count(dataset = clean,
                       cat_var = str_cols,
                       output_dir = output_vis_path + '\\Countplots'
                       )
    
    
    ################
    #-- Boxplots --#
    ################
    
    va.visualise.boxplot(dataset = clean,
                         num_var = ['logSalePrice'],
                         cat_var = ord_bin_cols,
                         output_dir = output_vis_path + '\\Boxplots'
                         )
    
    
    #####################
    #-- Scatter Plots --#
    #####################
    
    va.visualise.regplot(dataset = clean, 
                         res_var = ['logSalePrice'],
                         pred_var = num_cols,
                         output_dir = output_vis_path + '\\Scatterplots'
                         )
