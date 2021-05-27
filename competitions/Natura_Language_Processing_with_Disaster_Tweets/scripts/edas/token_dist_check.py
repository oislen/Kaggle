# -*- coding: utf-8 -*-
"""
Created on Thu May 27 11:17:49 2021

@author: oislen
"""

# import relevant libraries
import pandas as pd

def token_dist_check(data,
                     text_col,
                     n = 10
                     ):
    
    """
    
    Token Distribution Check Documentation
    
    Function Overview
    
    This function plots the a bar plot of the tokens for a given dataset and text column
    
    Defaults
    
    token_dist_check(data,
                     text_col,
                     n = 10
                     )
    
    Parameters
    
    data - DataFrame, the data to plot the tokens of the text_col with
    text_col - String, the column name of the text to plot in the data
    n - Integer, the number of levels to show from 1
    
    Returns
    
    tokens_series - Series, the token series use to create the plot from the data and text column
    
    Example
    
    token_dist_check(data = data,
                     text_col = 'text_norm_clean',
                     n = 10
                     )
    
    """
    
    # extract out all tokens seperated by spaces
    tokens = [word for tweet in data[text_col].to_list() for word in tweet.split(' ')]
    
    # count up all tokens
    tokens_series = pd.Series(tokens).value_counts()
    
    # token series counts distribution
    tokens_series_dist = tokens_series.value_counts()
    
    # plot a bar plot od the distribution
    tokens_series_dist.head(n).plot(kind = 'bar', title = text_col)
    
    return tokens_series
