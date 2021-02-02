# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 19:31:35 2019

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
report_01_raw_eda_subdir = static_inputs.loc['report_01_raw_eda', 'input']
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

print('loading in the raw data ...')

# import the raw data
test_identity, test_transaction, train_identity, train_transaction = prg.load_data(data_type = 'raw_data')

################
#-- Raw EDA  --#
################

print('Creating raw eda ...')

# create a list of the datasets
dataset_dict = {'test_identity.csv':test_identity, 
                'test_transaction.csv':test_transaction, 
                'train_identity.csv':train_identity, 
                'train_transaction.csv':train_transaction
                }

# create the output directory
output_report_dir = report_dir + report_01_raw_eda_subdir

# run the descriptive stats eda
prg.desc_stats(dataset_dict = dataset_dict, 
               output_report_dir = output_report_dir
               )