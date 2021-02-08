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
import cons

# load cusotm functions
import value_analysis as va

# load in data
base = pd.read_csv(cons.base_data_fpath, sep = '|')

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

"""
#############################
#-- Decriptive Statistics --#
#############################
"""

# generate numeric statistics
numeric_stats = va.Desc.stats(dataset = base, 
                              attrs = base.columns, 
                              stats_type = "numeric",
                              output_dir = cons.univar_stats_dir
                              )

string_stats = va.Desc.stats(dataset = base, 
                             attrs = base.columns, 
                             stats_type = "string",
                             output_dir = cons.univar_stats_dir
                             )

"""
######################
#-- Visualisations --#
######################
"""

##################
#-- Histograms --#
##################

va.Vis.hist(dataset = base,
            num_var = ['Age', 'Fare', 'FamSize', 'Parch', 'SibSp', 'Pclass'],
            bins = None,
            hist = True,
            kde = False
            )

###################
#-- Count Plots --#
###################

va.Vis.count(dataset = base,
             cat_var = ['cat_survived']
             )

va.Vis.count(dataset = base,
             cat_var = ['Sex']
             )

va.Vis.count(dataset = base,
             cat_var = ['cat_alone']
             )

va.Vis.count(dataset = base,
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

# generate the association tests
va.Desc.assoc(dataset = base,
              attrs = ['Survived', 'Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked'],
              response = None,
              assoc_type = 'nominal',
              output_dir = cons.bivar_assoc_dir
              )

#########################
#-- Correlation Tests --#
#########################


# generate the correlation tests
va.Desc.corr(dataset = base,
             attrs = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare'],
             response = None,
             corr_type = 'spearman',
             output_dir = cons.bivar_corr_dir
             )

# generate the correlation tests
va.Desc.corr(dataset = base,
             attrs = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare'],
             response = None,
             corr_type = 'kendall',
             output_dir = cons.bivar_corr_dir
             )

# generate the correlation tests
va.Desc.corr(dataset = base,
             attrs = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare'],
             response = None,
             corr_type = 'goodmans_g',
             output_dir = cons.bivar_corr_dir
             )

# generate the correlation tests
va.Desc.corr(dataset = base,
             attrs = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare'],
             response = None,
             corr_type = 'somers_d',
             output_dir = cons.bivar_corr_dir
             )

"""
######################
#-- Visualisations --#
######################
"""

###################
#-- Swarm Plots --#
###################

va.Vis.swarmplot(dataset = base,
                 num_var = ['Age'],
                 cat_var = ['cat_survived']
                 )

va.Vis.swarmplot(dataset = base,
                 num_var = ['Fare'],
                 cat_var = ['cat_survived']
                 )

va.Vis.swarmplot(dataset = base,
                 num_var = ['FamSize'],
                 cat_var = ['cat_survived']
                 )

va.Vis.swarmplot(dataset = base,
                 num_var = ['Parch'],
                 cat_var = ['cat_survived']
                 )

va.Vis.swarmplot(dataset = base,
                 num_var = ['SibSp'],
                 cat_var = ['cat_survived']
                 )

################
#-- Boxplots --#
################

va.Vis.boxplot(dataset = base,
               num_var = ['Age'],
               cat_var = ['cat_survived']
               )

va.Vis.boxplot(dataset = base,
               num_var = ['Fare'],
               cat_var = ['cat_survived']
               )

va.Vis.boxplot(dataset = base,
               num_var = ['FamSize'],
               cat_var = ['cat_survived']
               )

va.Vis.boxplot(dataset = base,
               num_var = ['Parch'],
               cat_var = ['cat_survived']
               )

va.Vis.boxplot(dataset = base,
               num_var = ['SibSp'],
               cat_var = ['cat_survived']
               )

####################
#-- Violin Plots --#
####################

va.Vis.violin(dataset = base,
              num_var = ['Age'],
              cat_var = ['cat_survived']
              )

va.Vis.violin(dataset = base,
              num_var = ['Fare'],
              cat_var = ['cat_survived']
              )

va.Vis.violin(dataset = base,
              num_var = ['FamSize'],
              cat_var = ['cat_survived']
              )

va.Vis.violin(dataset = base,
              num_var = ['Parch'],
              cat_var = ['cat_survived']
              )

va.Vis.violin(dataset = base,
              num_var = ['SibSp'],
              cat_var = ['cat_survived']
              )

###################
#-- Count Plots --#
###################

va.Vis.count(dataset = base,
             cat_var = ['cat_pclass'],
             hue_var = ['cat_survived']
             )

va.Vis.count(dataset = base,
             cat_var = ['Sex'],
             hue_var = ['cat_survived']
             )

va.Vis.count(dataset = base,
             cat_var = ['Embarked'],
             hue_var = ['cat_survived']
             )

####################
#-- Gains Charts --#
####################

#-- Age --#

# gains charts
va.MeasureGains.analysis(dataset = base,
                 attr = 'Age',
                 target = 'Survived',
                 bin_split = 'equal_width',
                 n_bins = 4,
                 title = 'Survival Rates by Age',
                 path = cons.bivar_gains_dir
                 )

#-- Class --#

# gains charts
va.MeasureGains.analysis(dataset = base,
                         attr = 'Pclass',
                         target = 'Survived',
                         bin_split = 'value',
                         title = 'Survival Rates by Class',
                         path = cons.bivar_gains_dir
                         )

#-- Sibblings and Spouses --#

va.MeasureGains.analysis(dataset = base,
                         attr = 'SibSp',
                         target = 'Survived',
                         bin_split = 'value',
                         title = 'Survival Rates by SibSp',
                         path = cons.bivar_gains_dir
                         )

#-- Parent and Child --#

va.MeasureGains.analysis(dataset = base,
                         attr = 'Parch',
                         target = 'Survived',
                         bin_split = 'value',
                         title = 'Survival Rates by Parch',
                         path = cons.bivar_gains_dir
                         )

va.MeasureGains.analysis(dataset = base,
                         attr = 'Age',
                         target = 'Fare',
                         bin_split = 'equal_size',
                         n_bins = 5,
                         title = 'Fare Rates by Age',
                         path = cons.bivar_gains_dir
                         )

va.MeasureGains.analysis(dataset = base,
                         attr = 'Pclass',
                         target = 'Fare',
                         bin_split = 'value',
                         title = 'Fare Rates by Class',
                         path = cons.bivar_gains_dir
                         )

#-- Target: Age --#

va.MeasureGains.analysis(dataset = base,
                         attr = 'Pclass',
                         target = 'Age',
                         bin_split = 'value',
                         title = 'Age Relativity by Class',
                         path = cons.bivar_gains_dir
                         )
    
va.MeasureGains.analysis(dataset = base,
                         attr = 'SibSp',
                         target = 'Age',
                         bin_split = 'value',
                         title = 'Age Relativity by SibSp',
                         path = cons.bivar_gains_dir
                         )


va.MeasureGains.analysis(dataset = base,
                         attr = 'Parch',
                         target = 'Age',
                         bin_split = 'value',
                         title = 'Age Relativity by Parch',
                         path = cons.bivar_gains_dir
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

va.Vis.scatter(dataset = base, 
               res_var = ['Fare'],
               pred_var = ['Age'],
               hue_var = ['cat_survived']
               )

va.Vis.swarmplot(dataset = base,
                 num_var = ['Fare'],
                 cat_var = ['cat_pclass'],
                 hue_var = ['cat_survived']
                 )

va.Vis.boxplot(dataset = base,
               num_var = ['FamSize'],
               cat_var = ['cat_survived'],
               hue_var = ['Sex']
               )

va.Vis.boxplot(dataset = base,
               num_var = ['Age'],
               cat_var = ['cat_survived'],
               hue_var = ['Sex']
               )

va.Vis.swarmplot(dataset = base,
                 num_var = ['Age'],
                 cat_var = ['cat_survived'],
                 hue_var = ['Sex']
                 )

####################
#-- Strip Plots --#
####################

va.Vis.stripplot(dataset = base,
                 num_var = ['Age'],
                 cat_var = ['cat_survived'],
                 hue_var = ['Sex']
                 )

###################
#-- Point Plots --#
###################

va.Vis.pointplot(dataset = base,
                 num_var = ['Age'],
                 cat_var = ['cat_survived'],
                 hue_var = ['Sex'],
                 estimator = np.mean
                 )

####################
#-- Violin Plots --#
####################

va.Vis.violin(dataset = base,
              num_var = ['Age'],
              cat_var = ['cat_survived'],
              hue_var = ['cat_pclass'],
              split = False
              )