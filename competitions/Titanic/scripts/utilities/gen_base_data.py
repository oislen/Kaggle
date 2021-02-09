# -*- coding: utf-8 -*-
"""
Created on Sat Nov  3 10:42:39 2018

@author: oislen
"""

# load in libraries
import pandas as pd
import numpy as np
import cons 

def gen_base_data(train_fpath,
                  test_fpath,
                  base_fpath
                  ):
    
    """
    
    Generate Base Data Documentation
    
    Function Overview
    
    This funciton generates the base titanic dataset by concatenating the training and test datasets.
    The function also performs a number of other data processing steps:
        * Adding indicator for train and test sets
        * Creating family size attribute
        * Creating alone attribute
    The data outputs the base data to the specified file path
    
    Defaults
    
    gen_base_data(train_fpath,
                  test_fpath,
                  base_fpath
                  )
    
    Parameters
    
    train_fpath - String, the input file path for the training data
    test_fpath - String, the input file path for the testing data
    base_fpath - String, the output file path for the concatenate base data
    
    Returns
    
    0 for successful execution
    
    Example
    
    gen_base_data(train_fpath = 'C:\\Users\\...\\train.csv',
                  test_fpath = 'C:\\Users\\...\\test.csv',
                  base_fpath = 'C:\\Users\\...\\base.csv'
                  )
    
    """

    # load in data
    train = pd.read_csv(train_fpath, sep = ',')
    test = pd.read_csv(test_fpath, sep = ',')

    print('Concatenating files ...')
    
    # note the training set as an extra column, the target survived
    train.columns
    test.columns
    
    # create a 'Survived' column in test
    test['Survived'] = np.nan
    
    # create a dataset indicator column
    train['Dataset'] = 'train'
    test['Dataset'] = 'test'
    
    # row bind the datasets
    base = pd.concat(objs = [train, test], axis = 0, sort = False)
    
    print('Engineering new features ...')
    
    # create a family size attribute
    base['FamSize'] = base['Parch'] + base['SibSp']
    
    # create an alone attribute
    base['Alone'] = (base['FamSize'] == 0).astype(int)

    print('Outputting base file ...')
    
    # output the dataset
    base.to_csv(base_fpath,
                   sep = '|',
                   encoding = 'utf-8',
                   header = True,
                   index = False
                   )
    
    return 0
