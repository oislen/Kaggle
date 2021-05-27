# -*- coding: utf-8 -*-
"""
Created on Tue May 25 10:27:59 2021

@author: oislen
"""
import os
import sys
sys.path.append(os.path.dirname(os.getcwd()))
import cons
import numpy as np
import pandas as pd
import spacy
import langdetect
import re
langdetect.DetectorFactory.seed = 0
# custom functions
from normalise_tweet import normalise_tweet
from spell_corrector import spell_corrector
from token_dist_check import token_dist_check

# TODO:
# create custom word embeddings with word2vec (gensim)
# GloVe (word vector embeddings)
# lots of text cleaning (nltk?)

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

# combine train and test
train['dataset'] = 'train'
test['target'] = np.nan
test['dataset'] = 'test'
data = pd.concat(objs = [train, test], ignore_index = True)

# determine which the languages of each tweet
# data['language'] = data['text'].apply(lambda x: langdetect.detect(x))

# run spell corrector
# note this is super slow due to for loop iterating over each word in each tweet!
# could alternatively create dictionary of incorrect_spelling:correct_spelling and map results acorss all strings?
# data['text_spell_checked'] = data['text'].apply(lambda x: spell_corrector(x, lang = 'en'))

# create spacy instance 
nlp = spacy.load('en_core_web_sm')

# run text normalisation function
data['text_norm'] = data['text'].apply(lambda x: normalise_tweet(tweet = x, nlp = nlp, norm_configs = cons.norm_configs))

# find lengths of normalised text
# some lengths are zero
data['text_norm_len'] = data['text_norm'].apply(lambda x: len(x))

# remove special @ characters
data['text_norm_clean'] = data['text_norm'].apply(lambda x: re.sub('@', '', x))

# plot the tokens distribution
token_dist_check(data = data, col ='text_norm_clean')

# extract out all tokens seperated by spaces
tokens = [word for tweet in data['text_norm_clean'].to_list() for word in tweet.split(' ')]
# count up all tokens
tokens_series = pd.Series(tokens).value_counts()
# examine tokens with 1 occuence
tokens_series_occur1 = tokens_series[tokens_series == 1]
tokens_series_occur1.head(25)

