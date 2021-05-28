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
comp_dir = os.path.join(root_dir, 'competitions\\Disaster_Tweets')
data_dir = os.path.join(comp_dir, 'data')
scripts_dir = os.path.join(comp_dir, 'scripts')
reports_dir = os.path.join(comp_dir, 'report')
models_dir = os.path.join(comp_dir, 'models')
utilities_nlp = os.path.join(scripts_dir, 'utilities')
arch_dir = os.path.join(scripts_dir, 'arch')
utilities_dir = os.path.join(root_dir, 'utilities')
utilities_comp = os.path.join(utilities_dir, 'comp')
utilities_graph = os.path.join(utilities_dir, 'graph')
utilities_model = os.path.join(utilities_dir, 'model')
utilties_preproc = os.path.join(utilities_dir, 'preproc')

# set data directories
raw_data_dir = os.path.join(data_dir, 'raw')
ref_data_dir = os.path.join(data_dir, 'ref')
# set raw data file names
raw_train_fname = 'train.csv'
raw_test_fname = 'test.csv'
raw_sampe_submissoin_fname = 'sample_submission.csv'
# set raw file paths
raw_train_fpath = os.path.join(raw_data_dir, raw_train_fname)
raw_test_fpath = os.path.join(raw_data_dir, raw_test_fname)
raw_sample_submission_fpath = os.path.join(raw_data_dir, raw_sampe_submissoin_fname)

# set the model predictions output file locaiton and name
pred_data_dir = os.path.join(data_dir, 'pred')
pred_fname = 'keras_rnn_preds.csv'
pred_fpath = os.path.join(pred_data_dir, pred_fname)

# append utilities directory to path
for p in [utilities_comp, utilities_graph, utilities_model, utilties_preproc, utilities_nlp]:
    sys.path.append(p)

# set url for glove wikipedia pretrained word vectors
glove_wiki_zip_url = 'http://nlp.stanford.edu/data/glove.6B.zip'
glove_wiki_zip_fname = glove_wiki_zip_url.split('/')[-1]
glove_wiki_zip_fpath = os.path.join(ref_data_dir, glove_wiki_zip_fname)
# set the names of the glove text files
glove_50d_fname = 'glove.6B.50d.txt'
glove_100d_fname = 'glove.6B.100d.txt'
glove_200d_fname = 'glove.6B.200d.txt'
glove_300d_fname = 'glove.6B.300d.txt'
# set the file paths of the glove text files
glove_50d_fpath = os.path.join(ref_data_dir, glove_50d_fname)
glove_100d_fpath = os.path.join(ref_data_dir, glove_100d_fname)
glove_200d_fpath = os.path.join(ref_data_dir, glove_200d_fname)
glove_300d_fpath = os.path.join(ref_data_dir, glove_300d_fname)
# set url for glove twitter pretrained word vectors
glove_twitter_zip_url = 'http://nlp.stanford.edu/data/glove.twitter.27B.zip'
glove_twitter_zip_fname = glove_twitter_zip_url.split('/')[-1]
glove_twitter_zip_fpath = os.path.join(ref_data_dir, glove_twitter_zip_fname)
# set the names of the glove text files
glove_twitter_25d_fname = 'glove.twitter.27B.25d.txt'
glove_twitter_50d_fname = 'glove.twitter.27B.50d.txt'
glove_twitter_100d_fname = 'glove.twitter.27B.100d.txt'
glove_twitter_200d_fname = 'glove.twitter.27B.200d.txt'
# set the file paths of the glove text files
glove_twitter_25d_fpath = os.path.join(ref_data_dir, glove_twitter_25d_fname)
glove_twitter_50d_fpath = os.path.join(ref_data_dir, glove_twitter_50d_fname)
glove_twitter_100d_fpath = os.path.join(ref_data_dir, glove_twitter_100d_fname)
glove_twitter_200d_fpath = os.path.join(ref_data_dir, glove_twitter_200d_fname)

##########################
#-- Cleaning Constants --#
##########################

# set normalisation constants
norm_configs = {'remove_bracket':True,
                'remove_currency':True,
                'remove_digit':True,
                'remove_email':True,
                'remove_num':True,
                'remove_punct':True,
                'remove_quote':True,
                'remove_stop':True,
                'remove_space':True,
                'remove_url':True,
                'to_lower':True
                }

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
