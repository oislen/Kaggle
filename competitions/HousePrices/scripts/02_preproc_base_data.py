# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 16:55:02 2021

@author: oislen
"""

# import relevant libraries
import cons
from preproc.create_base_data import create_base_data
from preproc.clean_base_data import clean_base_data
from preproc.engin_base_data import engin_base_data

def preproc_base_data(train_data_fpath,
                      test_data_fpath,
                      base_data_fpath,
                      clean_data_fpath,
                      engin_data_fpath
                      ):
    
    """
    
    Preprocess Base Data Documentation 
    
    Function Overview
    
    This function runs the data preprocessing for the base data set.
    This includes:
        1. Generating the base dataset by concatenating the train and test data sets
        2. Cleaning the base dataset
        3. Engineering new features from the cleaned base dataset
        
    Defaults
    
    preproc_base_data(train_data_fpath,
                      test_data_fpath,
                      base_data_fpath,
                      clean_data_fpath,
                      engin_data_fpath
                      )
    
    Parameters
    
    train_data_fpath - String, the file path to the training dataset
    test_data_fpath - String, the file path to the test dataset
    base_data_fpath - String, the file path to the base dataset
    clean_data_fpath - String, the file path to the cleaned base dataset
    engin_data_fpath - String, the file path to the engineered base dataset
    
    Returns
    
    0 for successful execution
    
    Example
    
    
    preproc_base_data(train_data_fpath = 'C:\\Users\\...\\train.csv',
                      test_data_fpath = 'C:\\Users\\...\\test.csv',
                      base_data_fpath = 'C:\\Users\\...\\base.csv',
                      clean_data_fpath = 'C:\\Users\...\\clean.csv',
                      engin_data_fpath = 'C:\\Users\\...\\engin.csv'
                      )    
    
    """
    
    print('~~~~~ Generating base data ...')
    
    # call function
    create_base_data(train_data_fpath = train_data_fpath,
                     test_data_fpath = test_data_fpath,
                     base_data_fpath = base_data_fpath
                     )

    print('~~~~~ Cleaning base data ...')
    
    # run clean base data
    clean_base_data(base_data_fpath = base_data_fpath,
                    clean_data_fpath = clean_data_fpath
                    )
    
    print('~~~~~ Engineering base data ...')
    
    # run data engineering script
    engin_base_data(clean_data_fpath = clean_data_fpath,
                    engin_data_fpath = engin_data_fpath
                    )
    
    
    return 0

if __name__ == '__main__':

    
    # extract file paths from cons
    train_data_fpath = cons.train_data_fpath
    test_data_fpath = cons.test_data_fpath
    base_data_fpath = cons.base_data_fpath
    clean_data_fpath = cons.clean_data_fpath
    engin_data_fpath = cons.engin_data_fpath
    
    # run data preprocessing
    preproc_base_data(train_data_fpath = train_data_fpath,
                      test_data_fpath = test_data_fpath,
                      base_data_fpath = base_data_fpath,
                      clean_data_fpath = clean_data_fpath,
                      engin_data_fpath = engin_data_fpath
                      )
    