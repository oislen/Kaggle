# -*- coding: utf-8 -*-
"""
Created on Tue May 25 10:27:59 2021

@author: oislen
"""
import os
import sys
sys.path.append(os.path.dirname(os.getcwd()))
import cons
import numpy as np
import pandas as pd
import spacy
import codecs
import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
from gensim.models import Phrases
from gensim.models.word2vec import LineSentence
from gensim.corpora import Dictionary, MmCorpus
from gensim.models.ldamulticore import LdaMulticore
from gensim.models import Word2Vec
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import CountVectorizer
# custom functions
from normalise_tweet import normalise_tweet

# load in the raw data files
train = pd.read_csv(cons.raw_train_fpath)
test = pd.read_csv(cons.raw_test_fpath)
sample_submission = pd.read_csv(cons.raw_sample_submission_fpath)

# dimensions of raw data files
train.shape
test.shape
sample_submission.shape

# columns within raw data files
train.columns
test.columns
sample_submission.columns

# view head of raw data files
train.head()
test.head()
sample_submission.head()

# number of mising values per column
train.isnull().sum()
test.isnull().sum()
sample_submission.isnull().sum()

# distinct values
train.keyword.value_counts()
train.target.value_counts()
train.location.value_counts()
train.text.value_counts()

# crosstabs
crosstab_keywords_target = pd.crosstab(index = train.keyword, columns = train.target)
crosstab_location_target = pd.crosstab(index = train.location, columns = train.target)
crosstab_text_target = pd.crosstab(index = train.text, columns = train.target)

# combine train and test
train['dataset'] = 'train'
test['target'] = np.nan
test['dataset'] = 'test'
data = pd.concat(objs = [train, test], ignore_index = True)

# create custom word embeddings with word2vec (gensim)
# GloVe (word vector embeddings)
# lots of text cleaning (nltk?)

# messing with spacy
nlp = spacy.load("en_core_web_sm")
parsed_tweet = nlp(data.text[3])
list(parsed_tweet)
list(parsed_tweet.sents)
list(parsed_tweet.ents)
# speech tagging
token_text = [token.orth_ for token in parsed_tweet]
token_pos = [token.pos_ for token in parsed_tweet]
pd.DataFrame(zip(token_text, token_pos), columns = ['token_text', 'part_of_speech'])
# text normalisation
token_lemma = [token.lemma_ for token in parsed_tweet]
token_shape = [token.shape_ for token in parsed_tweet]
pd.DataFrame(zip(token_text, token_lemma, token_shape), columns=['token_text', 'token_lemma', 'token_shape'])
# token lvel entity analysis
token_entity_type = [token.ent_type_ for token in parsed_tweet]
token_entity_iob = [token.ent_iob_ for token in parsed_tweet]
pd.DataFrame(zip(token_text, token_entity_type, token_entity_iob), columns = ['token_text', 'entity_type', 'inside_outside_begin'])
# relative frequency of tokens, and whether or not a token matches any of these categories
token_attributes = [(token.orth_, token.prob, token.is_stop, token.is_punct, token.is_space, token.like_num, token.is_oov) for token in parsed_tweet]
df = pd.DataFrame(token_attributes, columns = ['text', 'log_probability', 'stop?', 'punctuation?', 'whitespace?', 'number?', 'out of vocab.?'])
df.loc[:, 'stop?':'out of vocab.?'] = (df.loc[:, 'stop?':'out of vocab.?'].applymap(lambda x: u'Yes' if x else u''))                                               
df

# messing with gensim and phrase modelling
# cleaning in memeory with pandas dataframes
norm_configs = {'remove_bracket':True,
                'remove_currency':True,
                'remove_digit':True,
                'remove_email':True,
                'remove_num':True,
                'remove_punct':True,
                'remove_quote':True,
                'remove_stop':True,
                'remove_space':True,
                'remove_url':True
                }
# run text normalisation function
data['text_clean'] = data['text'].apply(lambda x: normalise_tweet(tweet = x, nlp = nlp, norm_configs = norm_configs))
#  bi-grams modelling
# create a stram of sentances for input into bi-gram model
sent_stream = [sent.split(' ') for sent in data['text_clean'].tolist()]
# train bigram model
bigram_model = Phrases(sent_stream)
# apply trained bi-gram model to formatted text
data['text_clean_bigram'] = data['text_clean'].apply(lambda x: ' '.join(bigram_model[x.split(' ')]))

# create a stram of sentances for corpus dict
sent_stream = [sent.split(' ') for sent in data['text_clean_bigram'].tolist()]
# topic model with LDA
id2word = Dictionary(sent_stream)
# Create Corpus
texts = sent_stream
# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]
# double check corpus and text lengths
len(corpus)
len(texts)
# number of topics
num_topics = 10
# Build LDA model
lda_model = LdaMulticore(corpus = corpus,
                         id2word = id2word,
                         num_topics = num_topics,
                         workers = 2
                         ) 
# define helper function to show lda topics
def explore_topic(topic_number, topn = 25):
    """
    accept a user-supplied topic number and
    print out a formatted list of the top terms
    """
    print(u'{:20} {}'.format(u'term', u'frequency') + u'\n')
    for term, frequency in lda_model.show_topic(topic_number, topn = 25):
        print(u'{:20} {:.3f}'.format(term, round(frequency, 3)))
# topics don't work well as tweets all relate to disasters
explore_topic(topic_number = 0)
explore_topic(topic_number = 1)
explore_topic(topic_number = 2)
explore_topic(topic_number = 3)
# describe tweets
list(lda_model[corpus])[0]
texts
# define helper function to predict topics
def pred_topic(string_input):
    string_input_idx = texts.index(string_input.split(' '))
    string_input_corpus = corpus[string_input_idx]
    preds = lda_model[string_input_corpus]
    preds_out = [tup[1] for tup in preds]
    return preds_out
# predict ofr single string
string_input = data['text_clean_bigram'][1]
pred_topic(string_input)
# predict for all strings
string_preds_list = {val:pred_topic(val) for idx, val in enumerate(data['text_clean_bigram'])}
# convert output to df
string_preds_df = pd.DataFrame.from_dict(string_preds_list, orient = 'index')
string_preds_df.reset_index().rename(columns = {'index':'text'})

# playing with word2vec
# create a stram of sentances for word 2 vec
sent_stream = [sent.split(' ') for sent in data['text_clean_bigram'].tolist()]
# initiate the model and perform the first epoch of training
tweet2vec = Word2Vec(sent_stream, vector_size=100, window=5, min_count=20, sg=1, workers=2, epochs = 5, negative = 10)
# perform another 10 epochs of training
for i in range(10):
    tweet2vec.train(sent_stream, total_examples = tweet2vec.corpus_count, epochs = tweet2vec.epochs)
tweet2vec.wv['storm']
tweet2vec.wv.most_similar(positive = 'storm')
tweet2vec.wv.similarity("storm", "hurricane")
tweet2vec.wv['hurricane']
print(u'{:,} terms in the food2vec vocabulary.'.format(len(tweet2vec.wv.index_to_key)))
# build a list of the terms, integer indices, and term counts from the food2vec model vocabulary
ordered_vocab = [(term, tweet2vec.wv.get_index(term, term), tweet2vec.wv.get_vecattr(term, "count"), tweet2vec.wv.get_vector(term)) for term in tweet2vec.wv.index_to_key]
# sort by the term counts, so the most common terms appear first
#ordered_vocab = sorted(ordered_vocab, key = lambda (term, index, count): -count)
# unzip the terms, integer indices, and counts into separate lists
ordered_terms, term_indices, term_counts, word_vecs = zip(*ordered_vocab)
word_vecs_dict = dict(zip(ordered_terms, word_vecs))
# create a DataFrame with the food2vec vectors as data, and the terms as row labels
word_vectors = pd.DataFrame.from_dict(data = word_vecs_dict, orient = 'index')
# define helper function for printing similar tokens
def get_related_terms(token, topn=10):
    """
    look up the topn most similar terms to token
    and print them as a formatted list
    """
    for word, similarity in tweet2vec.wv.most_similar(positive=[token], topn=topn):
        print(u'{:20} {}'.format(word, round(similarity, 3)))
get_related_terms(u'storm')
# define helper function for word algebra
def word_algebra(add=[], subtract=[], topn=1):
    """
    combine the vectors associated with the words provided
    in add= and subtract=, look up the topn most similar
    terms to the combined vector, and print the result(s)
    """
    answers = tweet2vec.wv.most_similar(positive=add, negative=subtract, topn=topn)
    for term, similarity in answers:
        print(term)
# take the violence out of storm you get california
word_algebra(add=[u'storm'], subtract=[u'violent'])
# plot with tsne
# create tsne object
tsne = TSNE()
# fit and transform data
tsne_vectors = tsne.fit_transform(word_vectors.values)
# convert output to dataframe
tsne_vectors_df = pd.DataFrame(tsne_vectors, index = word_vectors.index, columns = [u'x_coord', u'y_coord'])
# plot results as a scatterplot
tsne_vectors_df.head()
tsne_vectors_df.plot.scatter(x = 'x_coord', y = 'y_coord')
