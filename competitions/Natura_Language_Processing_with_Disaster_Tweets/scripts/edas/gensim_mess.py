# -*- coding: utf-8 -*-
"""
Created on Wed May 26 20:09:50 2021

@author: oislen
"""

# load in relevant libraries
import os
import sys
sys.path.append(os.path.dirname(os.getcwd()))
import cons
import re
import spacy
import pandas as pd
import numpy as np
from gensim.models import Phrases
from gensim.corpora import Dictionary
from gensim.models.ldamulticore import LdaMulticore
from gensim.models import Word2Vec
from sklearn.manifold import TSNE
from normalise_tweet import normalise_tweet
from lda_topic_prob import lda_topic_prob
from topic_df import topic_df

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm
from keras.models import Sequential
from keras.layers import Embedding,LSTM,Dense,SpatialDropout1D
from keras.initializers import Constant
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
from nltk.tokenize import word_tokenize
from tqdm import tqdm

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

# create spacy instance 
nlp = spacy.load('en_core_web_sm')
# run text normalisation function
data['text_norm'] = data['text'].apply(lambda x: normalise_tweet(tweet = x, nlp = nlp, norm_configs = cons.norm_configs))
# remove special @ characters
# could replace cases with @user?
data['text_norm_clean'] = data['text_norm'].apply(lambda x: re.sub('@', '', x))

################################
#-- Bi-gram Phrase Modelling --#
################################

# create a stram of sentances for input into bi-gram model
sent_stream = [sent.split(' ') for sent in data['text_norm_clean'].tolist()]
# train bigram model
bigram_model = Phrases(sent_stream)
# apply trained bi-gram model to formatted text
data['text_clean_bigram'] = data['text_norm_clean'].apply(lambda x: ' '.join(bigram_model[x.split(' ')]))

###########################
#-- LDA Topic Modelling --#
###########################

# create a stram of sentances for corpus dict
input_data = [sent.split(' ') for sent in data['text_clean_bigram'].tolist()]
# topic model with LDA
id2word = Dictionary(input_data)
# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in input_data]
# number of topics
num_topics = 10
# Build LDA model
lda_model = LdaMulticore(corpus = corpus, id2word = id2word, num_topics = num_topics, workers = 2) 

# topics don't work well as tweets all relate to disasters
lda_model.show_topic(topicid = 0, topn = 25)
lda_model.show_topic(topicid = 1, topn = 25)
lda_model.show_topic(topicid = 2, topn = 25)
lda_model.show_topic(topicid = 3, topn = 25)

# predict of single string
string_input = data['text_clean_bigram'][1]
lda_topic_prob(string_input, input_data, lda_model)

# predict for all strings
# takes ~20 minutes to run
data['lda_topic_prob'] = data['text_clean_bigram'].apply(lambda x: lda_topic_prob(x, input_data, lda_model))
# transform the lda topic probabilities into a dataframe representation
# takes a while to run
lda_topic_df = topic_df(data = data, prob_col = 'lda_topic_prob', text_col = 'text_clean_bigram', lda_model = lda_model)

###########################
#-- Word2Vec Embeddings --#
###########################

# create a stram of sentances for word 2 vec
sent_stream = [sent.split(' ') for sent in data['text_norm_clean'].tolist()]
# initiate the model and perform the first epoch of training
tweet2vec = Word2Vec(sent_stream, vector_size = 100, window = 3, min_count = 1, sg = 1)

# check word vectors for storm
tweet2vec.wv['storm']
# find the most similar word vector
tweet2vec.wv.most_similar(positive = 'storm')
# find out how similar storm and hurricane are
tweet2vec.wv.similarity("storm", "hurricane")
# word algebra
tweet2vec.wv.most_similar(positive = ['storm'], negative = ['violent'])

# total number of word vectors
len(tweet2vec.wv.index_to_key)
# build a list of the terms, integer indices, and term counts from the food2vec model vocabulary
ordered_vocab = [(term, tweet2vec.wv.get_index(term, term), tweet2vec.wv.get_vecattr(term, "count"), tweet2vec.wv.get_vector(term)) for term in tweet2vec.wv.index_to_key]
# unzip the terms, integer indices, and counts into separate lists
ordered_terms, term_indices, term_counts, word_vecs = zip(*ordered_vocab)
# create a dictionary of terms and associated word vectors
word_vecs_dict = dict(zip(ordered_terms, word_vecs))
# create a DataFrame with the tweet2vec vectors as data, and the terms as row labels
word_vectors = pd.DataFrame.from_dict(data = word_vecs_dict, orient = 'index')

# plot with tsne
# create tsne object
tsne = TSNE()
# fit and transform data
tsne_vectors = tsne.fit_transform(word_vectors.values)
# convert output to dataframe
tsne_vectors_df = pd.DataFrame(tsne_vectors, index = word_vectors.index, columns = [u'x_coord', u'y_coord'])
# plot results as a scatterplot
tsne_vectors_df.plot.scatter(x = 'x_coord', y = 'y_coord')

###################
#-- Keras Model --#
###################


# create training data
# note bi-grams negatively affected in word2vec
# create corpus object
embedding_dict = word_vecs_dict
corpus = [word for sent in data['text_norm_clean'].to_list() for word in sent.split() if word in word_vecs_dict.keys()]
MAX_LEN = 50
tokenizer_obj = Tokenizer()
tokenizer_obj.fit_on_texts(corpus)
sequences = tokenizer_obj.texts_to_sequences(corpus)
tweet_pad = pad_sequences(sequences,maxlen = MAX_LEN, truncating = 'post', padding = 'post')
word_index = tokenizer_obj.word_index
num_words=len(word_index)+1
embedding_matrix=np.zeros((num_words,100))
for word,i in tqdm(word_index.items()):
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
test_df=tweet_pad[train.shape[0]:]
X_train,X_test,y_train,y_test=train_test_split(train_df,train['target'].values,test_size=0.15)
history=model.fit(X_train,y_train,batch_size=4,epochs=15,validation_data=(X_test,y_test),verbose=2)

# make test predictions
y_pre=model.predict(test)
y_pre=np.round(y_pre).astype(int).reshape(3263)
sub=pd.DataFrame({'id':sample_submission['id'].values.tolist(),'target':y_pre})