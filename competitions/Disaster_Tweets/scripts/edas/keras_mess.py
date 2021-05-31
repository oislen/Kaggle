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
import spacy
# custom functions
import shahules_utils as shutils
from load_glove_word_vecs import load_glove_word_vecs
from normalise_tweet import normalise_tweet

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
#from nltk.tokenize import spacy


from rnn_model import rnn_model
from fit_model import fit_model
from keras.callbacks import ReduceLROnPlateau

#################
#-- Data Prep --#
#################

# load in the raw data files
train = pd.read_csv(cons.raw_train_fpath)
test = pd.read_csv(cons.raw_test_fpath)
sample_submission = pd.read_csv(cons.raw_sample_submission_fpath)

# combine train and test
train['dataset'] = 'train'
test['target'] = np.nan
test['dataset'] = 'test'
data = pd.concat(objs = [train, test], ignore_index = True)

# if running shahules data prep steps
if False:
    
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
    corpus_v2 = [sent.split() for sent in data['text_norm'].to_list()]

###################
#-- Keras Model --#
###################

# load in glove word vectors
# create embedding dictionary
glove_dims = 100
embedding_dict = load_glove_word_vecs(cons.glove_100d_fpath)

# create training data
# note bi-grams negatively affected in word2vec
max_len = 50
tokenizer_obj = Tokenizer()
tokenizer_obj.fit_on_texts(corpus_v2)
sequences = tokenizer_obj.texts_to_sequences(corpus_v2)
tweet_pad = pad_sequences(sequences, maxlen = max_len, truncating = 'post', padding = 'post')
word_index = tokenizer_obj.word_index
num_words=len(word_index)+1
embedding_matrix=np.zeros((num_words,glove_dims))
for word,i in word_index.items():
    if i > num_words:
        continue
    emb_vec=embedding_dict.get(word)
    if emb_vec is not None:
        embedding_matrix[i]=emb_vec

# generate the training data
train_df = tweet_pad[:train.shape[0]]

# split training data into training and vlaidation sets
X_train, X_valid, y_train, y_valid = train_test_split(train_df,
                                                      train['target'].values,
                                                      test_size = 0.15
                                                      )



# generate rnn model architecture
model = rnn_model(embedding_matrix = embedding_matrix,
                  input_length = max_len
                  )

# define optimiser and compile
optimizer = Adam(learning_rate = 1e-5)

# set a learning rate annealer
learning_rate_reduction = ReduceLROnPlateau(monitor = 'val_accuracy', 
                                            patience = 3, 
                                            verbose = 1, 
                                            factor = 0.5, 
                                            min_lr = 0.00001
                                            )

# Attention: Windows implementation may cause an error here. In that case use model_name=None.
fit_model(model = model, 
          epochs = 5,
          starting_epoch = None,
          batch_size = None,
          valid_batch_size = None,
          output_dir = cons.checkpoints_dir,
          optimizer = optimizer,
          metric = 'accuracy',
          loss = 'binary_crossentropy',
          lrate_red = learning_rate_reduction,
          X_train = X_train,
          X_val = X_valid, 
          Y_train = y_train, 
          Y_val = y_valid
          )

# make test predictions
test_df=tweet_pad[train.shape[0]:]
y_pre=model.predict(test_df)
y_pre = np.round(y_pre).astype(int).reshape(3263)
sub=pd.DataFrame({'id':sample_submission['id'].values.tolist(),'target':y_pre})
sub.to_csv(cons.pred_fpath, index=False)

