# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 07:24:17 2021

@author: oislen
"""

import cons
import pandas as pd
import numpy as np
import spacy
import shahules_utils as shutils
from normalise_tweet import normalise_tweet

def process_data(shahules):
    
    # load in the raw data files
    train = pd.read_csv(cons.raw_train_fpath)
    test = pd.read_csv(cons.raw_test_fpath)
    
    # combine train and test
    train['dataset'] = 'train'
    test['target'] = np.nan
    test['dataset'] = 'test'
    data = pd.concat(objs = [train, test], ignore_index = True)
    
    # if running shahules data prep steps
    if shahules:
        
        # apply data cleaning
        data['text'] = data['text'].apply(lambda x : shutils.remove_URL(x))
        data['text'] = data['text'].apply(lambda x : shutils.remove_html(x))
        data['text'] = data['text'].apply(lambda x: shutils.remove_emoji(x))
        data['text'] = data['text'].apply(lambda x : shutils.remove_punct(x))
        
        # create corpus object given available word vectors
        corpus = shutils.create_corpus(df = data, text_col = 'text')
    
    # using custom data prep steps
    else:
        
        # create spacy instance 
        nlp = spacy.load('en_core_web_sm')
        
        # run text normalisation function
        data['text_norm'] = data['text'].apply(lambda x: normalise_tweet(tweet = x, nlp = nlp, norm_configs = cons.norm_configs))
        
        # remove special @ characters
        # could replace cases with @user?
        #data['text_norm_clean'] = data['text_norm'].apply(lambda x: re.sub('@', '', x))
    
        # create a corpus to feed into the model
        corpus = [sent.split() for sent in data['text_norm'].to_list()]
        
    return corpus, data