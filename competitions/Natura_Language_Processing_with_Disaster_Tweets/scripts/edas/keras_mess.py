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
import re
import pandas as pd
import numpy as np
import spacy
# custom functions
import shahules_utils as shutils
from load_glove_word_vecs import load_glove_word_vecs
from normalise_tweet import normalise_tweet

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding,LSTM,Dense,SpatialDropout1D
from keras.initializers import Constant
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
#from nltk.tokenize import spacy

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
embedding_dict = load_glove_word_vecs(cons.glove_100d_fpath)

# create training data
# note bi-grams negatively affected in word2vec
MAX_LEN = 50
tokenizer_obj = Tokenizer()
tokenizer_obj.fit_on_texts(corpus_v2)
sequences = tokenizer_obj.texts_to_sequences(corpus_v2)
tweet_pad = pad_sequences(sequences, maxlen = MAX_LEN, truncating = 'post', padding = 'post')
word_index = tokenizer_obj.word_index
num_words=len(word_index)+1
embedding_matrix=np.zeros((num_words,100))
for word,i in word_index.items():
    if i > num_words:
        continue
    emb_vec=embedding_dict.get(word)
    if emb_vec is not None:
        embedding_matrix[i]=emb_vec

# fit model
model=Sequential()
embedding=Embedding(num_words,100,embeddings_initializer=Constant(embedding_matrix),input_length=MAX_LEN,trainable=False)
model.add(embedding)
model.add(SpatialDropout1D(0.2))
model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))
optimzer=Adam(learning_rate=1e-5)
model.compile(loss='binary_crossentropy',optimizer=optimzer,metrics=['accuracy'])
model.summary()
train_df=tweet_pad[:train.shape[0]]
X_train,X_test,y_train,y_test=train_test_split(train_df,train['target'].values,test_size=0.15)
history=model.fit(X_train,y_train,batch_size=4,epochs=15,validation_data=(X_test,y_test),verbose=2)

# make test predictions
test_df=tweet_pad[train.shape[0]:]
y_pre=model.predict(test_df)
y_pre = np.round(y_pre).astype(int).reshape(3263)
sub=pd.DataFrame({'id':sample_submission['id'].values.tolist(),'target':y_pre})
sub.to_csv(cons.pred_fpath, index=False)
