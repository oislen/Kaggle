# -*- coding: utf-8 -*-
"""
Created on Wed May 26 13:00:42 2021

@author: oislen
"""

# define normalisation function
def normalise_tweet(tweet, 
                    nlp, 
                    norm_configs, 
                    lowercase = True
                    ):
    
    """
    
    Normalise Tweet Documentation
    
    Function Overview
    
    This function applies text normalisation to a given tweet
    
    Defaults
    
    normalise_tweet(tweet, 
                    nlp, 
                    norm_configs, 
                    lowercase = True
                    )
    Parameters

    tweet - String, the text to normalise
    nlp - Spacy Instance, the spacy object to use for the normalisation
    norm_configs - Dictionary, the normalisation configurations
    lowercase - Boolean, whether to standardise text to lower case, default is True
    
    Returns
    
    norm_tweet - String, the normalised text
    
    Example
    
    # define the normalisation configurations
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
    
    # call the normalisation function
    normalise_tweet(tweet = 'orest fire near La Ronge Sask. Canada', 
                    nlp = spacy.load("en_core_web_sm"), 
                    norm_configs = norm_configs,
                    lowercase = True
                    )
        
    """
    
    # if converting raw tweet to lower case
    if lowercase:
        # standardise to lower case
        tweet = tweet.lower()
        
    # parse tweet with spacy
    parsed_tweet = nlp(tweet)
    
    # create list to hold normalised lemmas
    lemmatized = list()
    
    # iterate through each word in the parsed tweet
    for word in parsed_tweet:
        
        # if removing stop words
        if norm_configs['remove_stop'] and word.is_stop:
            continue
        
        # if removing quotations marks
        if norm_configs['remove_quote'] and word.is_quote:
            continue
        
        # if removing punctuation
        if norm_configs['remove_punct'] and word.is_punct:
            continue
        
        # if removing brackets
        if norm_configs['remove_bracket'] and word.is_bracket:
            continue
        
        # if removing urls
        if norm_configs['remove_url'] and word.like_url:
            continue
        
        # if removing emails
        if norm_configs['remove_email'] and word.like_email:
            continue
        
        # if removing currency
        if norm_configs['remove_currency'] and word.is_currency:
            continue
        
        # if removing digit
        if norm_configs['remove_digit'] and word.is_digit:
            continue
        
        # if removing numbers
        if norm_configs['remove_num'] and word.like_num:
            continue
        
        # if removing spaces
        if norm_configs['remove_space'] and word.is_space:
            continue
        
        # otherwise extract normalised word lemma
        else:
            
            # extract the normalised word lemma
            lemmatized.append(word.lemma_)
            
    # recombined normalised tweet
    norm_tweet = " ".join(lemmatized)
    
    return norm_tweet