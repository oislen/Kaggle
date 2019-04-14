# -*- coding: utf-8 -*-
"""
Created on Sat Nov  3 11:21:09 2018

@author: oislen
"""

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
sys.path.append('C:/Users/User/Documents/Data_Analytics/Python/value_analysis')
import value_analysis as va

# load in data
input_dir = 'C:\\Users\\User\\Documents\\Kaggle\\Titanic\\data\\attempt_3\\'
base_name = 'base.csv'
base = pd.read_csv(input_dir + base_name, sep = '|')

# define the general output dir
report_dir = 'C:\\Users\\User\\Documents\\Kaggle\\Titanic\\report\\attempt_3\\'

# redefine some of the numeric variables into categories
base['cat_survived'] = base['Survived'].map({1.0:'yes', 0.0:'no'})
base['cat_pclass'] = base['Pclass'].astype(str)
base['cat_sibsp'] = base['SibSp'].astype(str)
base['cat_parch'] = base['Parch'].astype(str)
base['cat_famsize'] = base['FamSize'].astype(str)
base['cat_alone'] = base['Alone'].astype(str)

"""
###############################################################################
#-- Univariate Analysis --#####################################################
###############################################################################

This section of the script performs a univariate analysis.

"""

# define the output path for the visualisations
output_vis_path = report_dir + 'univariate_analysis\\plots\\'

"""
#############################
#-- Decriptive Statistics --#
#############################
"""

# define the output path for the descript statistics
output_desc_stats_path = report_dir + 'univariate_analysis\\descriptive_stats\\'

# generate numeric statistics
numeric_stats = va.desc_stats(dataset = base, 
                              attrs = base.columns, 
                              stats_type = "numeric",
                              path = output_desc_stats_path
                              )

string_stats = va.desc_stats(dataset = base, 
                             attrs = base.columns, 
                             stats_type = "string",
                             path = output_desc_stats_path
                             )

"""
######################
#-- Visualisations --#
######################
"""

##################
#-- Histograms --#
##################

va.vis_hist(dataset = base,
            num_var = ['Age', 'Fare', 'FamSize', 'Parch', 'SibSp', 'Pclass'],
            bins = None,
            hist = True,
            kde = False
            )

###################
#-- Count Plots --#
###################

va.vis_count(dataset = base,
             cat_var = ['cat_survived']
             )

va.vis_count(dataset = base,
             cat_var = ['Sex']
             )

va.vis_count(dataset = base,
             cat_var = ['cat_alone']
             )

va.vis_count(dataset = base,
             cat_var = ['Embarked']
             )

"""
###############################################################################
#-- Bivariate Analysis --######################################################
###############################################################################

This section of the script performs a bivariate analysis 

"""

"""
#############################
#-- Decriptive Statistics --#
#############################
"""

#########################
#-- Association Tests --#
#########################

# define the output path for the association tests
output_assoc_path = report_dir + 'bivariate_analysis\\association_tests\\'

# generate the association tests
va.var_assoc(dataset = base,
             attrs = ['Survived', 'Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked'],
             response = None,
             assoc_type = 'nominal',
             path = output_assoc_path
             )

#########################
#-- Correlation Tests --#
#########################

# define the output path for the association tests
output_cor_path = report_dir + 'bivariate_analysis\\correlation_tests\\'

# generate the correlation tests
va.var_corr(dataset = base,
            attrs = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare'],
            response = None,
            corr_type = 'spearman',
            path = output_cor_path
            )

# generate the correlation tests
va.var_corr(dataset = base,
            attrs = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare'],
            response = None,
            corr_type = 'kendall',
            path = output_cor_path
            )

# generate the correlation tests
va.var_corr(dataset = base,
            attrs = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare'],
            response = None,
            corr_type = 'goodmans_g',
            path = output_cor_path
            )

# generate the correlation tests
va.var_corr(dataset = base,
            attrs = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare'],
            response = None,
            corr_type = 'somers_d',
            path = output_cor_path
            )

"""
######################
#-- Visualisations --#
######################
"""

###################
#-- Swarm Plots --#
###################

va.vis_swarmplot(dataset = base,
                 num_var = ['Age'],
                 cat_var = ['cat_survived']
                 )

va.vis_swarmplot(dataset = base,
                 num_var = ['Fare'],
                 cat_var = ['cat_survived']
                 )

va.vis_swarmplot(dataset = base,
                 num_var = ['FamSize'],
                 cat_var = ['cat_survived']
                 )

va.vis_swarmplot(dataset = base,
                 num_var = ['Parch'],
                 cat_var = ['cat_survived']
                 )

va.vis_swarmplot(dataset = base,
                 num_var = ['SibSp'],
                 cat_var = ['cat_survived']
                 )

################
#-- Boxplots --#
################

va.vis_boxplot(dataset = base,
               num_var = ['Age'],
               cat_var = ['cat_survived']
               )

va.vis_boxplot(dataset = base,
               num_var = ['Fare'],
               cat_var = ['cat_survived']
               )

va.vis_boxplot(dataset = base,
               num_var = ['FamSize'],
               cat_var = ['cat_survived']
               )

va.vis_boxplot(dataset = base,
               num_var = ['Parch'],
               cat_var = ['cat_survived']
               )

va.vis_boxplot(dataset = base,
               num_var = ['SibSp'],
               cat_var = ['cat_survived']
               )

####################
#-- Violin Plots --#
####################

va.vis_violin(dataset = base,
              num_var = ['Age'],
              cat_var = ['cat_survived']
              )

va.vis_violin(dataset = base,
              num_var = ['Fare'],
              cat_var = ['cat_survived']
              )

va.vis_violin(dataset = base,
              num_var = ['FamSize'],
              cat_var = ['cat_survived']
              )

va.vis_violin(dataset = base,
              num_var = ['Parch'],
              cat_var = ['cat_survived']
              )

va.vis_violin(dataset = base,
              num_var = ['SibSp'],
              cat_var = ['cat_survived']
              )

###################
#-- Count Plots --#
###################

va.vis_count(dataset = base,
             cat_var = ['cat_pclass'],
             hue_var = ['cat_survived']
             )

va.vis_count(dataset = base,
             cat_var = ['Sex'],
             hue_var = ['cat_survived']
             )

va.vis_count(dataset = base,
             cat_var = ['Embarked'],
             hue_var = ['cat_survived']
             )

####################
#-- Gains Charts --#
####################

# define the output path for the descript statistics
output_gains_path = report_dir + 'bivariate_analysis\\measure_gains\\'

#-- Age --#

# gains charts
va.measure_gains(dataset = base,
                 attr = 'Age',
                 target = 'Survived',
                 bin_split = 'equal_width',
                 n_bins = 4,
                 title = 'Survival Rates by Age',
                 path = output_gains_path
                 )

#-- Class --#

# gains charts
va.measure_gains(dataset = base,
                 attr = 'Pclass',
                 target = 'Survived',
                 bin_split = 'value',
                 title = 'Survival Rates by Class',
                 path = output_gains_path
                 )

#-- Sibblings and Spouses --#

va.measure_gains(dataset = base,
                 attr = 'SibSp',
                 target = 'Survived',
                 bin_split = 'value',
                 title = 'Survival Rates by SibSp',
                 path = output_gains_path
                 )

#-- Parent and Child --#

va.measure_gains(dataset = base,
                 attr = 'Parch',
                 target = 'Survived',
                 bin_split = 'value',
                 title = 'Survival Rates by Parch',
                 path = output_gains_path
                 )









va.measure_gains(dataset = base,
                 attr = 'Age',
                 target = 'Fare',
                 bin_split = 'equal_size',
                 n_bins = 5,
                 title = 'Fare Rates by Age',
                 path = output_gains_path
                 )

va.measure_gains(dataset = base,
                 attr = 'Pclass',
                 target = 'Fare',
                 bin_split = 'value',
                 title = 'Fare Rates by Class',
                 path = output_gains_path
                 )

#-- Target: Age --#

va.measure_gains(dataset = base,
                 attr = 'Pclass',
                 target = 'Age',
                 bin_split = 'value',
                 title = 'Age Relativity by Class',
                 path = output_gains_path
                 )

va.measure_gains(dataset = base,
                 attr = 'SibSp',
                 target = 'Age',
                 bin_split = 'value',
                 title = 'Age Relativity by SibSp',
                 path = output_gains_path
                 )


va.measure_gains(dataset = base,
                 attr = 'Parch',
                 target = 'Age',
                 bin_split = 'value',
                 title = 'Age Relativity by Parch',
                 path = output_gains_path
                 )

"""
###########################
#-- Trivariate Analysis --#
###########################

This section of the script performs a multivariate analysis 

"""

"""
######################
#-- Visualisations --#
######################
"""

#####################
#-- Scatter Plots --#
#####################

va.vis_scatter(dataset = base, 
               res_var = ['Fare'],
               pred_var = ['Age'],
               hue_var = ['cat_survived']
               )

va.vis_swarmplot(dataset = base,
                 num_var = ['Fare'],
                 cat_var = ['cat_pclass'],
                 hue_var = ['cat_survived']
                 )

va.vis_boxplot(dataset = base,
               num_var = ['FamSize'],
               cat_var = ['cat_survived'],
               hue_var = ['Sex']
               )

va.vis_boxplot(dataset = base,
               num_var = ['Age'],
               cat_var = ['cat_survived'],
               hue_var = ['Sex']
               )

va.vis_swarmplot(dataset = base,
                 num_var = ['Age'],
                 cat_var = ['cat_survived'],
                 hue_var = ['Sex']
                 )

####################
#-- Strip Plots --#
####################

va.vis_stripplot(dataset = base,
                 num_var = ['Age'],
                 cat_var = ['cat_survived'],
                 hue_var = ['Sex']
                 )

###################
#-- Point Plots --#
###################

va.vis_pointplot(dataset = base,
                 num_var = ['Age'],
                 cat_var = ['cat_survived'],
                 hue_var = ['Sex'],
                 estimator = np.mean
                 )

####################
#-- Violin Plots --#
####################

va.vis_violin(dataset = base,
              num_var = ['Age'],
              cat_var = ['cat_survived'],
              hue_var = ['cat_pclass'],
              split = False
              )