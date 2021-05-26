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
import spacy
import codecs
import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
from gensim.models import Phrases
from gensim.models.word2vec import LineSentence
# custom functions
from normalise_tweet import normalise_tweet

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

# messing with spacy
nlp = spacy.load("en_core_web_sm")
parsed_tweet = nlp(train.text[3])
list(parsed_tweet)
list(parsed_tweet.sents)
list(parsed_tweet.ents)
# speech tagging
token_text = [token.orth_ for token in parsed_tweet]
token_pos = [token.pos_ for token in parsed_tweet]
pd.DataFrame(zip(token_text, token_pos), columns = ['token_text', 'part_of_speech'])
# text normalisation
token_lemma = [token.lemma_ for token in parsed_tweet]
token_shape = [token.shape_ for token in parsed_tweet]
pd.DataFrame(zip(token_text, token_lemma, token_shape), columns=['token_text', 'token_lemma', 'token_shape'])
# token lvel entity analysis
token_entity_type = [token.ent_type_ for token in parsed_tweet]
token_entity_iob = [token.ent_iob_ for token in parsed_tweet]
pd.DataFrame(zip(token_text, token_entity_type, token_entity_iob), columns = ['token_text', 'entity_type', 'inside_outside_begin'])
# relative frequency of tokens, and whether or not a token matches any of these categories
token_attributes = [(token.orth_, token.prob, token.is_stop, token.is_punct, token.is_space, token.like_num, token.is_oov) for token in parsed_tweet]
df = pd.DataFrame(token_attributes, columns = ['text', 'log_probability', 'stop?', 'punctuation?', 'whitespace?', 'number?', 'out of vocab.?'])
df.loc[:, 'stop?':'out of vocab.?'] = (df.loc[:, 'stop?':'out of vocab.?'].applymap(lambda x: u'Yes' if x else u''))                                               
df

# messing with gensim and phrase modelling
# cleaning in memeory with pandas dataframes
norm_configs = {'remove_bracket':True,
                'remove_currency':True,
                'remove_digit':True,
                'remove_email':True,
                'remove_num':True,
                'remove_punct':True,
                'remove_quote':True,
                'remove_stop':True,
                'remove_space':True,
                'remove_url':True
                }

# run text normalisation function
train['text_clean'] = train['text'].apply(lambda x: normalise_tweet(tweet = x, nlp = nlp, norm_configs = norm_configs))
