# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 12:22:34 2021

@author: oislen
"""

# load relevant libraries
import os
import pandas as pd
import cons
from sklearn.model_selection import train_test_split

def process_data(train_data_fpath,
                 test_data_fpath,
                 valid_size = 0.1,
                 random_state = 1234
                 ):
    
    """
    
    Process Data Documentation
    
    Function Overview
    
    This function loads and process the training and testing data for the digit competition.
    
    Once the data is loaded the following data processing occurs:
        * splitting features and target from training data
        * normalising features
        * dummy encoding targets
        * splitting into training and validation sets
    
    Defaults 
    
    process_data(train_data_fpath,
                 test_data_fpath,
                 test_size = 0.1,
                 random_state = 1234
                 )
    
    Parameters
    
    train_data_fpath - String, the full input file path to the training data
    test_data_fpath - String, the full input file path to the test data
    test_size - Float, the proportion of data in the validation set, default is 0.1
    random_state - Integer, the random seed, default is 1234
    
    Returns
    
    X_train - Array, the processed training features
    y_train - Array, the processed training target
    X_valid - Array, the processed validation features
    y_valid - Array, the processed validation target
    X_test - Array, the processed test features
    
    Example
    
    process_data(train_data_fpath = 'C:\\Users\\...\\train.csv',
                 test_data_fpath = 'C:\\Users\\...\\test.csv',
                 valid_size = 0.1,
                 random_state = 1234
                 )
    
    """
    
    print('checking inputs ...')
    
    # check for string data types
    str_types = [train_data_fpath, test_data_fpath]
    if any([type(str_inp) != str for str_inp in str_types]):
        raise TypeError('Input Type Error: the input parameters [train_data_fpath, test_data_fpath] must be string data types.')
        
    # check for float data types
    float_types = [valid_size]
    if any([type(float_inp) != float for float_inp in float_types]):
        raise TypeError('Input Type Error: the input parameters [valid_size] must be float data types.')
    
    # check for integer data types
    int_types = [random_state]
    if any([type(int_inp) != int for int_inp in int_types]):
        raise TypeError('Input Type Error: the input parameters [random_state] must be integer data types.')
    
    # check file paths exist
    if os.path.exists(train_data_fpath) == False:
        raise OSError('File Error: the input train file does not exist {}.'.format(train_data_fpath))
    if os.path.exists(test_data_fpath) == False:
        raise OSError('File Error: the input test file does not exist {}.'.format(test_data_fpath))
        
    
    print('loading in data ...')
    
    # load in train and test data
    train = pd.read_csv(cons.train_data_fpath)
    test = pd.read_csv(cons.test_data_fpath)
    
    print('splitting features and target from training data ...')
    
    # extract out target and features from training data
    y_train = train['label']
    X_train = train.drop(columns = ['label'])
    
    print('normalising features ...')
    
    # normalise training data
    X_train = X_train / 255
    X_test = test.values / 255
    
    print('dummy encoding targets ...')
    
    # dummy encode labels
    y_train = pd.get_dummies(y_train)
    
    print('splitting into training and validation sets ...')
    
    # randomly split the data into training and validation sets
    X_train, X_valid, y_train, y_valid = train_test_split(X_train.values, 
                                                          y_train.values,
                                                          test_size = valid_size, 
                                                          random_state = random_state
                                                          )
    
    return X_train, y_train, X_valid, y_valid, X_test

# run process function
X_train, y_train, X_valid, y_valid, X_test = process_data(train_data_fpath = cons.train_data_fpath,
                                                          test_data_fpath = cons.test_data_fpath,
                                                          valid_size = cons.valid_size,
                                                          random_state = cons.random_state
                                                          )