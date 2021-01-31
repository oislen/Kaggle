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

# set directories
root_dir = 'C:\\Users\\User\\Documents\\GitHub\\Kaggle\\Digit_Recognizer'
scripts_dir = os.path.join(root_dir, 'scripts')
data_dir = os.path.join(root_dir, 'data')

# define filenames
zip_data_fname = '{}.zip'.format(comp_name)

# create file paths
zip_data_fpath = os.path.join(data_dir, zip_data_fname)