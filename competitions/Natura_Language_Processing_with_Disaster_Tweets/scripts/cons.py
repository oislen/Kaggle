# -*- coding: utf-8 -*-
"""
Created on Tue May 25 10:11:01 2021

@author: oislen
"""

# import relevant libraries
import os
import sys
import pandas as pd

# set pandas options
pd.set_option('display.max_columns', 10)

# set competition name
comp_name = 'nlp-getting-started'
download_data = True
unzip_data = True
del_zip = True

# set project directories
git_dir = 'C:\\Users\\User\\Documents\\GitHub'
#git_dir = '/run'
root_dir = os.path.join(git_dir, 'Kaggle')
comp_dir = os.path.join(root_dir, 'competitions\\Natura_Language_Processing_with_Disaster_Tweets')
data_dir = os.path.join(comp_dir, 'data')
scripts_dir = os.path.join(comp_dir, 'scripts')
reports_dir = os.path.join(comp_dir, 'report')
models_dir = os.path.join(comp_dir, 'models')
utilities_dir = os.path.join(root_dir, 'utilities')
utilities_comp = os.path.join(utilities_dir, 'comp')
utilities_graph = os.path.join(utilities_dir, 'graph')
utilities_model = os.path.join(utilities_dir, 'model')
utilties_preproc = os.path.join(utilities_dir, 'preproc')

# set data directories
raw_data_dir = os.path.join(data_dir, 'raw')


# append utilities directory to path
for p in [utilities_comp, utilities_graph, utilities_model, utilties_preproc]:
    sys.path.append(p)

######################
#-- Plot Constants --#
######################

plot_size_width = 12
plot_size_height = 8
plot_title_size = 25
plot_axis_text_size = 20
plot_label_size = 'x-large'
bins = 100
kde = False
