# -*- coding: utf-8 -*-
"""
Created on Thu May 27 13:35:16 2021

@author: oislen
"""

import pandas as pd

def topic_df(data,
             text_col,
             prob_col,
             lda_model
             ):
    
    """
    
    Topic DataFrame Documentation
    
    Function Overview
    
    This function generates a topic probability representaiton of a given lda model andtopic probabilities
    
    Defaults
    
    topic_df(data,
             prob_col,
             lda_model
             )
    
    Parameters
    
    data - DataFrame, the data with the topic probabilities to generate the topic dataframe with
    text_col - String, the column name of the original text used to generate the topic probabilities
    prob_col - String, the column name of the topic probabilities in the input data
    lda_model - models.ldamulticore.LdaMulticore, the lda model used to generate the topic probabilities
    
    Returns
    
    lda_topic_df - DataFrame, a topic dataframe representation of the topic probabilities
    
    Example
    
    topic_df(data = data,
             text_col = 'text_clean_bigram',
             prob_col = 'lda_topic_prob',
             lda_model = lda_model
             )
    
    """
    
    # extract the number of topics from the lda model
    num_topics = lda_model.num_topics
    
    # extract the index of the input data
    df_index = data.index
    
    # define the column names of the dataframe
    col_names = [text_col] + ['topic_{topic_num}'.format(topic_num = topic_num) for topic_num in range(num_topics)]
    
    # create a new output dataframe with the same index and topic ids as columns
    lda_topic_df = pd.DataFrame(index = df_index, columns = col_names)
    
    # assign the input text column data
    lda_topic_df[text_col] = data[text_col]
    
    # iterate through the input index
    for idx in df_index:
        
        # extract row
        row = data[prob_col][idx]
        
        # iterate through the probability tuples
        for tup in row:
            
            # create the topic column name
            tup_col_name = 'topic_{topic_num}'.format(topic_num = str(tup[0]))
            
            # assign the probability to the corresponding idx and topic column
            lda_topic_df.loc[idx, [tup_col_name]] = tup[1]
            
    # return the transformed lda topic df
    return lda_topic_df