# -*- coding: utf-8 -*-
"""
Created on Sun Nov 18 12:23:52 2018

@author: oislen
"""

# load in libraries
import os
import pandas as pd
import numpy as np
import cons 

def create_base(train_data_fpath,
                test_data_fpath,
                base_data_fpath
                ):
    
    """
    
    Create Base Data Documentation
    
    Function Overview
    
    Defaults
    
    Parameters
    
    Returns
    
    Example
    
    """
    
    print('Checking inputs ...')
    
    # check the string input parameter
    str_params = [train_data_fpath, test_data_fpath, base_data_fpath]
    if any([type(param) != str for param in str_params]):
        raise TypeError('Input parameters [train_data_fpath, test_data_fpath, base_data_fpath] must be string data types.')
    # check training file exists
    if os.path.exists(train_data_fpath) == False:
        raise OSError('Input train file path {} does not exist'.format(train_data_fpath))
    # check test file exists
    if os.path.exists(test_data_fpath) == False:
        raise OSError('Input test file path {} does not exist.'.format(test_data_fpath))
        
    print('Loading train and test data ...')
    
    # load the train and test data
    train_data = pd.read_csv(train_data_fpath, sep = cons.sep)
    test_data = pd.read_csv(test_data_fpath, sep = cons.sep)
    
    print('Concatenating files ...')
    
    # create a 'SalePrice' column in test
    test_data['SalePrice'] = np.nan
    
    # create a dataset indicator column
    train_data['Dataset'] = 'train'
    test_data['Dataset'] = 'test'
    
    # create the list of concate objects
    cat_obs = [train_data, test_data]
    
    # row bind the datasets
    base_data = pd.concat(objs = cat_obs, 
                          axis = 0, 
                          sort = False, 
                          ignore_index = True
                          )
    
    print('Outputting base file ...')
    
    # output the dataset
    base_data.to_csv(base_data_fpath,
                     sep = cons.sep,
                     encoding = cons.encoding,
                     header = cons.header,
                     index = cons.index
                     )
    
    return 0

# if running script as main programme
if __name__ == '__main__':
    
    # extract file paths from cons
    train_data_fpath = cons.train_data_fpath
    test_data_fpath = cons.test_data_fpath
    base_data_fpath = cons.base_data_fpath
    
    # call function
    create_base(train_data_fpath = train_data_fpath,
                test_data_fpath = test_data_fpath,
                base_data_fpath = base_data_fpath
                )