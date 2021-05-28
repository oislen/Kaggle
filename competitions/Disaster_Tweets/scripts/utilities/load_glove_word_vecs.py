# -*- coding: utf-8 -*-
"""
Created on Fri May 28 08:56:22 2021

@author: oislen
"""

# import relevant libraries
import numpy as np

def load_glove_word_vecs(glove_fpath, 
                         encoding = 'utf8'
                         ):
    
    """
    
    Load Glove Word Vectors Documentation
    
    Function Overview
    
    This function loads in pretrained glove word vectors given a specified glove .txt file
    
    https://nlp.stanford.edu/projects/glove/
    
    Defaults
    
    load_glove_word_vecs(glove_fpath, 
                         encoding = 'utf8'
                         )
    
    Parameters
    
    glove_fpath - String, the full input file path to the pretrained glove text file
    encoding - String, the encoding to use when reading in the glove text file, default is 'utf8'
    
    Returns
    
    embedding_dict - Dictionary, the word vectors
    
    Example
    
    load_glove_word_vecs(glove_fpath = 'C:\\Users\\...\\glove.twitter.27B.100d.txt', 
                         encoding = 'utf8'
                         )
    
    """
    
    # create an empty dictionary to hold the word embeddings
    embedding_dict = {}
    
    # open the text file
    with open(file = glove_fpath, mode = 'r', encoding = encoding) as f:
        
        # for each line in the file
        for line in f:
            
            # split line into list of words
            values = line.split()
            
            # extract the index word
            word = values[0]
            
            # extract the word vector
            vectors = np.asarray(values[1:],'float32')
            
            # assign to the embeddings dictioary
            embedding_dict[word] = vectors
            
    # close the text file
    f.close()
    
    return embedding_dict