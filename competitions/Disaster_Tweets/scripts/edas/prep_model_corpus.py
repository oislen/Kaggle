# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 20:09:04 2021

@author: oislen
"""
from tqdm import tqdm
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np

def prep_model_corpus(corpus, embedding_dict):
    
    """
    Prepare Model Corpus
    """
    
    # calculate the maximum corpus length
    max_len = max([len(tweet) for tweet in corpus])
    
    # calculate the number of dimensions to the word vector embeddings
    word_vec_dims = list(embedding_dict.values())[0].shape[0]
    
    # tokensize corpuses using indices
    tokenizer_obj = Tokenizer()
    tokenizer_obj.fit_on_texts(corpus)
    sequences = tokenizer_obj.texts_to_sequences(corpus)
    
    # add padding to tokenize sequence indices
    tweet_pad = pad_sequences(sequences, maxlen = max_len, truncating = 'post', padding = 'post')
    
    # extract the word inex
    word_index = tokenizer_obj.word_index
    
    # calculate the number of words
    num_words = len(word_index) + 1
    
    # create an empty matrix to hold 
    embedding_matrix = np.zeros((num_words, word_vec_dims))
    
    # iterate across each word and index
    for word, idx in tqdm(word_index.items()):

        # idx surpasses number of words
        if idx > num_words:
            
            continue
        
        # extract out the corresponding word vector
        emb_vec = embedding_dict.get(word)
        
        # if the embedded word vector is not empty
        if emb_vec is not None:
            
            # assign word vector to embedding matrix
            embedding_matrix[idx] = emb_vec
            
    return tweet_pad, embedding_matrix, max_len