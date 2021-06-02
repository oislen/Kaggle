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
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam

# custom functions
from load_glove_word_vecs import load_glove_word_vecs
from prep_model_corpus import prep_model_corpus
from process_data import process_data
from rnn_model import rnn_model
from fit_model import fit_model

# run data procressing
corpus, data = process_data(shahules = True)

# load in glove word vectors
# create embedding dictionary
embedding_dict, data = load_glove_word_vecs(cons.glove_100d_fpath)

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
          epochs = 15,
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

# make test predictions
sample_submission = pd.read_csv(cons.raw_sample_submission_fpath)
test_df=tweet_pad[train.shape[0]:]
y_pre=model.predict(test_df)
y_pre = np.round(y_pre).astype(int).reshape(3263)
sub=pd.DataFrame({'id':sample_submission['id'].values.tolist(),'target':y_pre})
sub.to_csv(cons.pred_fpath, index=False)