# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 10:17:01 2021

@author: oislen
"""

import os
import sys
scripts_dir = os.path.dirname(os.getcwd())
sys.path.append(scripts_dir)

import cons
import pandas as pd
import urllib
import re

# download ground truth labels
urllib.request.urlretrieve(cons.url, cons.ground_truth_fpath)

# load in ground truth, test and sample submission
test_data_with_labels = pd.read_csv(cons.ground_truth_fpath)
test_data = pd.read_csv(cons.test_data_fpath)
submission = pd.read_csv(cons.sample_sub_data_fpath)

# clean up name labels in ground truth
for i, name in enumerate(test_data_with_labels['name']):
    if '"' in name:
        test_data_with_labels['name'][i] = re.sub('"', '', name)
        
# clean up name labels in test file
for i, name in enumerate(test_data['Name']):
    if '"' in name:
        test_data['Name'][i] = re.sub('"', '', name)

# create empty list to hold survived info
survived = []

# append survived info
for name in test_data['Name']:
    survived.append(int(test_data_with_labels.loc[test_data_with_labels['name'] == name]['survived'].values[-1]))

# create submission file
submission['Survived'] = survived