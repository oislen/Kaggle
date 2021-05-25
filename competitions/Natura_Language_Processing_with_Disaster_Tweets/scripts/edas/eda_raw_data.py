# -*- coding: utf-8 -*-
"""
Created on Tue May 25 10:27:59 2021

@author: oislen
"""
import os
import sys
sys.path.append(os.path.dirname(os.getcwd()))
import cons
import pandas as pd

# load in the raw data files
train = pd.read_csv(cons.raw_train_fpath)
test = pd.read_csv(cons.raw_test_fpath)
sample_submission = pd.read_csv(cons.raw_sample_submission_fpath)

# dimensions of raw data files
train.shape
test.shape
sample_submission.shape

# columns within raw data files
train.columns
test.columns
sample_submission.columns

# view head of raw data files
train.head()
test.head()
sample_submission.head()

# number of mising values per column
train.isnull().sum()
test.isnull().sum()
sample_submission.isnull().sum()

# distinct values
train.keyword.value_counts()
train.target.value_counts()
train.location.value_counts()
train.text.value_counts()

# crosstabs
crosstab_keywords_target = pd.crosstab(index = train.keyword, columns = train.target)
crosstab_location_target = pd.crosstab(index = train.location, columns = train.target)
crosstab_text_target = pd.crosstab(index = train.text, columns = train.target)

# create custom word embeddings with word2vec (gensim)
# GloVe (word vector embeddings)
# lots of text cleaning (nltk?)