# -*- coding: utf-8 -*-
"""
Created on Mon May 31 12:54:32 2021

@author: oislen
"""

# load in relevant libraries
from keras.models import Sequential
from keras.layers import Embedding, Dense, LSTM,SpatialDropout1D
from keras.initializers import Constant

def rnn_model(embedding_matrix,
              input_length
              ):
    
    """
       
    Recurrent Neural Netwrok Model Documentation
    
    Function OVerview
    
    This function generates the RNN model archecture for the training data.
    
    Defaults
    
    rnn_model(embedding_matrix,
              input_length
              )
    
    Parameters
    
    embedding_matrix - Numpy Array, the embedding matrix of the input corpus contain word vector embeddings
    input_length - Integer, the maximum length of an input tweet
    
    Returns
    
    model - Keras Model, the RNN model archecture
    
    Example
    
    rnn_model(embedding_matrix = embedding_matrix,
              input_length = 30
              )
    
    """
    
    # set the dropout rate
    dropout_rate = 0.2
    
    # initiate the sequence model
    model = Sequential()
    
    # create an embeddings layer
    embeddings = Embedding(input_dim = embedding_matrix.shape[0],
                           output_dim = embedding_matrix.shape[1],
                           embeddings_initializer = Constant(embedding_matrix),
                           input_length = input_length,
                           trainable = False
                           )
    # add the embeddings to the sequence model
    model.add(embeddings)
    
    # create a spatial dropout layer
    spatial_dropout = SpatialDropout1D(dropout_rate)
    # add the spatial dropout to the sequence model
    model.add(spatial_dropout)
    
    # create an LSTM stage
    lstm = LSTM(units = 64, 
                dropout = dropout_rate, 
                recurrent_dropout = dropout_rate
                )
    # add lstm to sequence model
    model.add(lstm)
    
    # create a dense layer
    dense = Dense(units = 1, 
                  activation = 'sigmoid'
                  )
    # add dense layer to squence model
    model.add(dense)
    
    return model