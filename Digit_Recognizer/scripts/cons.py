# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 11:09:57 2021

@author: oislen
"""

# load libraries
import os

# set programme constants
comp_name = 'digit-recognizer'
download_data = True
unzip_data = True
del_zip = True
random_state = 1234
valid_size = 0.1

# set directories
root_dir = 'C:\\Users\\User\\Documents\\GitHub\\Kaggle\\Digit_Recognizer'
scripts_dir = os.path.join(root_dir, 'scripts')
data_dir = os.path.join(root_dir, 'data')

# define filenames
zip_data_fname = '{}.zip'.format(comp_name)
sample_sub_data_fname = 'sample_submission.csv'
test_data_fname = 'test.csv'
train_data_fname = 'train.csv'

# create file paths
zip_data_fpath = os.path.join(data_dir, zip_data_fname)
sample_sub_data_fpath = os.path.join(data_dir, sample_sub_data_fname)
test_data_fpath = os.path.join(data_dir, test_data_fname)
train_data_fpath = os.path.join(data_dir, train_data_fname)

