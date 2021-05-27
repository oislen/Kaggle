# -*- coding: utf-8 -*-
"""
Created on Wed May 26 20:05:22 2021

@author: oislen
"""

# import required libraries
import os
import sys
sys.path.append(os.path.dirname(os.getcwd()))
import spacy
import cons
import pandas as pd
import numpy as np
from normalise_tweet import normalise_tweet

# load in the raw data files
train = pd.read_csv(cons.raw_train_fpath)
test = pd.read_csv(cons.raw_test_fpath)
sample_submission = pd.read_csv(cons.raw_sample_submission_fpath)

# combine train and test
train['dataset'] = 'train'
test['target'] = np.nan
test['dataset'] = 'test'
data = pd.concat(objs = [train, test], ignore_index = True)

# create spacy instance
nlp = spacy.load("en_core_web_sm")

# test parsing a text string
parsed_tweet = nlp(data.text[3])
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

# token level entity analysis
token_entity_type = [token.ent_type_ for token in parsed_tweet]
token_entity_iob = [token.ent_iob_ for token in parsed_tweet]
pd.DataFrame(zip(token_text, token_entity_type, token_entity_iob), columns = ['token_text', 'entity_type', 'inside_outside_begin'])

# relative frequency of tokens, and whether or not a token matches any of these categories
token_attributes = [(token.orth_, token.prob, token.is_stop, token.is_punct, token.is_space, token.like_num, token.is_oov) for token in parsed_tweet]
df = pd.DataFrame(token_attributes, columns = ['text', 'log_probability', 'stop?', 'punctuation?', 'whitespace?', 'number?', 'out of vocab.?'])
df.loc[:, 'stop?':'out of vocab.?'] = (df.loc[:, 'stop?':'out of vocab.?'].applymap(lambda x: u'Yes' if x else u''))                                               
df

# configure spacy text processing
text = 'What if?!'
# set normalisation constants
norm_configs = {'remove_bracket':True,
                'remove_currency':True,
                'remove_digit':True,
                'remove_email':True,
                'remove_num':True,
                'remove_punct':True,
                'remove_quote':True,
                'remove_stop':False,
                'remove_space':True,
                'remove_url':True,
                'to_lower':True
                }
# run cleaning
normalise_tweet(tweet = text, norm_configs = norm_configs)
