# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 14:34:21 2019

@author: oislen
"""


# import relevant libraries
import pandas as pd
import importlib
import sys

#####################
#-- Preliminaries --#
#####################

print('Loading in packages ...')

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
report_subdir = static_inputs.loc['report_subdir', 'input']
report_02_full_eda_subdir = static_inputs.loc['report_02_full_eda', 'input']
scripts_subdir = static_inputs.loc['scripts_subdir', 'input']

# create the various directories
data_dir = project_dir + data_subdir
report_dir = project_dir + report_subdir
scripts_dir = project_dir + scripts_subdir

# load the import project function
sys.path.append(scripts_dir)
prg = importlib.import_module('prg_cis_fraud')

##################
#-- Load Data  --#
##################

# import the raw data
data_full = prg.load_data(data_type = 'full_data')

#################
#-- Full EDA  --#
#################

print('Creating full eda ...')

# create a list of the datasets
dataset_dict = {'data_full.csv':data_full}

# create the output directory
output_report_dir = report_dir + report_02_full_eda_subdir

# run the descriptive stats eda
prg.desc_stats(dataset_dict = dataset_dict, 
               output_report_dir = output_report_dir
               )

# run correlation statistics
prg.corr_stats(dataset_dict = dataset_dict, 
               output_report_dir = output_report_dir
               )
