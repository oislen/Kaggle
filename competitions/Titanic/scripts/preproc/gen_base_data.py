# -*- coding: utf-8 -*-
"""
Created on Sat Nov  3 10:42:39 2018

@author: oislen
"""

# load in libraries
import pandas as pd
import numpy as np
import os 
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
    
    print('checking inputs ...')
    
    # check input data types
    str_inputs = [train_fpath, test_fpath, base_fpath]
    if any([type(val) != str for val in str_inputs]):
        raise ValueError('Input params [train_fpath, test_fpath, base_fpath] must be str data types')
    # check if input file path exists
    if os.path.exists(train_fpath) == False:
        raise OSError('Input file path {} does not exist'.format(train_fpath))
    # check if output file path exists
    if os.path.exists(test_fpath) == False:
        raise OSError('Input file path {} does not exist'.format(test_fpath))
    
    print('loading in data ...')
    
    # load in data
    train = pd.read_csv(train_fpath, sep = cons.sep)
    test = pd.read_csv(test_fpath, sep = cons.sep)

    print('Concatenating files ...')

    # create a 'Survived' column in test
    test[cons.y_col[0]] = np.nan
    
    # create a dataset indicator column
    train['Dataset'] = 'train'
    test['Dataset'] = 'test'
    
    # create list of objects to concatenate
    concat_objs = [train, test]
    
    # row bind the datasets
    base = pd.concat(objs = concat_objs, 
                     axis = 0, 
                     ignore_index = True,
                     sort = False
                     )

    print('Outputting base file ...')
    
    # output the dataset
    base.to_csv(base_fpath,
                sep = cons.sep,
                encoding = cons.encoding,
                header = cons.header,
                index = cons.index
                )
    
    return 0
