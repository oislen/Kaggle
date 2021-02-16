# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 09:49:33 2021

@author: oislen
"""

# load in relevant libraries
import cons
from preproc.gen_base_data import gen_base_data
from preproc.clean_base_data import clean_base_data
from preproc.feat_engineer import feat_engineer

def prg_preproc_data(train_fpath,
                     test_fpath,
                     base_fpath,
                     base_clean_fpath,
                     base_engin_fpath
                     ):
    
    """
    
    Programme Preprocess Data Documenation
    
    Function Overview
    
    This function calls the data processing utility functions to clean and prep the raw data for modelling
    There are three parts to the preprocessing programme:
        1. Generate the base data
        2. Clean the base data
        3. Feature engineer the clea base data
    
    Defaults
    
    prg_preproc_data(train_fpath,
                     test_fpath,
                     base_fpath,
                     base_clean_fpath,
                     base_engin_fpath
                     )
    
    Parameters
    
    train_fpath - String, the file path to the training data
    test_fpath - String, the file path to the test data
    base_fpath - String, the file path to the base data
    base_clean_fpath - String, the file path to the cleaned base data
    base_engin_fpath - String, the file path to the engineered base data
    
    Returns
    
    0 for successful execution
    
    Example
    
    prg_preproc_data()
        
    """
    
    print('Checking inputs ...')
    
    # check input data types
    str_inputs = [train_fpath, test_fpath, base_fpath, base_clean_fpath, base_engin_fpath]
    if any([type(val) != str for val in str_inputs]):
        raise ValueError('Input params [train_fpath, test_fpath, base_fpath] must be str data types')
        
    print('~~~~~ Running base data generator ...')
    
  
    # generate base data
    gen_base_data(train_fpath = train_fpath,
                  test_fpath = test_fpath,
                  base_fpath = base_fpath
                  )
    
    print('~~~~~ Running base data cleaner ...')
    
    # generate clean base data
    clean_base_data(base_fpath = base_fpath,
                    base_clean_fpath = base_clean_fpath
                    )
    
    print('~~~~~ Running feature engineer ...')
    
    # engineer new features
    feat_engineer(base_clean_2_fpath = base_clean_fpath,
                  base_engin_fpath = base_engin_fpath
                  )
    
    return 0
    
if __name__ == '__main__':
    
    # extract file paths from cons.py
    train_fpath = cons.train_data_fpath
    test_fpath = cons.test_data_fpath
    base_fpath = cons.base_data_fpath
    base_clean_fpath = cons.base_clean_data_fpath
    base_engin_fpath = cons.base_engin_data_fpath
    
    # run preprocessing programme
    prg_preproc_data(train_fpath = train_fpath,
                     test_fpath = test_fpath,
                     base_fpath = base_fpath,
                     base_clean_fpath = base_clean_fpath,
                     base_engin_fpath = base_engin_fpath
                     )