# -*- coding: utf-8 -*-
"""
Created on Wed May 26 22:41:30 2021

@author: oislen
"""

# load in relevant libraries
from gensim.corpora import Dictionary

# define helper function to predict topics
def lda_topic_prob(string_input, 
                   input_data, 
                   lda_model
                   ):
    """
    
    Latent Dirichlet Allocation Topic Probabilities Documentation
    
    Function Overview
    
    This function generates the lda topic probabilities for a string given it's associated lda model and training data
    
    Defaults
    
    lda_topic_prob(string_input, 
                   input_data, 
                   lda_model
                   )
    
    Parameters
    
    string_input - String, the input string to generate lda topic probabilities for
    input_data - Series, the text data used to train the lda model that contains the input string
    lda_model - models.ldamulticore.LdaMulticore, the lda model trained with the input data that contains the input string 
    
    Returns
    
    preds_out - List, the string input probabilities associate for each topic
    
    """
    
    # create the id2word dictionary using the input data
    id2word = Dictionary(input_data)
    
    # recreate corpus using sent_stream
    corpus = [id2word.doc2bow(text) for text in input_data]
    
    # find correspond index of string in sentance steam 
    string_input_idx = input_data.index(string_input.split(' '))
    
    # extract out the corpus corresponding to the sentence stream
    string_input_corpus = corpus[string_input_idx]
    
    # make a prediction for the string using it's corpus entry
    preds = lda_model[string_input_corpus]
    
    # extract out the probabilities associate with each topic
    preds_out = [tup[1] for tup in preds]
    
    return preds_out