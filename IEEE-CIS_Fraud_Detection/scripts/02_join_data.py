# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 11:59:54 2019

@author: oislen
"""
# import relevant libraries
import pandas as pd
import importlib
import sys

#####################
#-- Preliminaries --#
#####################

print('Loading in packages and data ...')

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
report_01_intial_eda_subdir = static_inputs.loc['report_01_inital_eda', 'input']
scripts_subdir = static_inputs.loc['scripts_subdir', 'input']
python_dir  = static_inputs.loc['python_dir', 'input']
value_analysis_subdir  = static_inputs.loc['value_analysis_subdir', 'input']

# create the various directories
data_dir = project_dir + data_subdir
scripts_dir = project_dir + scripts_subdir

# load the import raw data function
sys.path.append(scripts_dir)
lrd = importlib.import_module('prg_cis_fraud')

# import the raw data
test_identity, test_transaction, train_identity, train_transaction = lrd.load_data(data_type = 'raw_data')

#########################
#-- Join the Datasets --#
#########################

print('joining test data ...')

# join the datasets
test_full = pd.merge(left = test_identity, 
                     right = test_transaction,
                     on = 'TransactionID',
                     how = 'outer'
                     )

print('joining train data ...')

# join the datasets
train_full = pd.merge(left = train_identity, 
                      right = train_transaction,
                      on = 'TransactionID',
                      how = 'outer'
                      )

print('assigning data identifer ...')

# assign the overall dataset identifier
test_full['dataset_id'] = 'test'
train_full['dataset_id'] = 'train'

################################
#-- Concatenate the Datasets --#
################################

print('concatenating datasets ...')

# concatenate the train and test datasets together
data_all = pd.concat(objs = [train_full, test_full],
                     axis = 0,
                     sort = False,
                     ignore_index = True
                     )

print('Outputting datasets ...')

# output the joined dataset
output_path = data_dir + 'data_all.csv'
data_all.to_csv(output_path,
                sep = ',',
                header = True,
                index = False,
                #compression  = 'gzip',
                encoding = 'latin1'
                )
