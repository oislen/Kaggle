# -*- coding: utf-8 -*-
"""
Created on Fri May 28 10:29:23 2021
@author: oislen
"""

# load in relevant libraries
import os
import sys
sys.path.append(os.path.dirname(os.getcwd()))
import cons
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam

# custom functions
from load_glove_word_vecs import load_glove_word_vecs
from prep_model_corpus import prep_model_corpus
from process_data import process_data
from rnn_model import rnn_model
from fit_model import fit_model
from gen_model_preds import gen_model_preds

# run data procressing
corpus, data = process_data(shahules = True)

# load in glove word vectors
# create embedding dictionary
embedding_dict = load_glove_word_vecs(cons.glove_100d_fpath)

# split data into train and test sets
train_filter = data['dataset'] == 'train'
train = data.loc[train_filter, :]

# run prep model corpus
tweet_pad, embedding_matrix, max_len = prep_model_corpus(corpus = corpus, 
                                                         embedding_dict = embedding_dict
                                                         )

# generate the training data
train_df = tweet_pad[:train.shape[0]]

# split training data into training and vlaidation sets
X_train, X_valid, y_train, y_valid = train_test_split(train_df,
                                                      train['target'].values,
                                                      test_size = 0.15,
                                                      random_state = 1
                                                      )

# generate rnn model architecture
model = rnn_model(embedding_matrix = embedding_matrix,
                  input_length = max_len
                  )

# define optimiser and compile
optimizer = Adam(learning_rate = 1e-5)

# Attention: Windows implementation may cause an error here. In that case use model_name=None.
fit_model(model = model, 
          epochs = 2,
          starting_epoch = None,
          batch_size = 4,
          valid_batch_size = None,
          output_dir = cons.checkpoints_dir,
          optimizer = optimizer,
          metric = 'accuracy',
          loss = 'binary_crossentropy',
          X_train = X_train,
          X_val = X_valid, 
          Y_train = y_train, 
          Y_val = y_valid
          )

# extract out the test data for model predictions
test_df = tweet_pad[train.shape[0]:]

# make the model predictions
sub_df = gen_model_preds(test_df = test_df, model = model)

# view head of submission file
sub_df.head()
