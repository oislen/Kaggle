# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 09:49:33 2021

@author: oislen
"""

import cons
from utilities.gen_base_data import gen_base_data
from utilities.clean_base_data import clean_base_data
from utilities.feat_engineer import feat_engineer

def prg_preproc_data():
    
    """
    
    Programme Preprocess Data Documenation
    
    Function Overview
    
    This function calls the data processing utility functions to clean and prep the raw data for modelling
    
    Defaults
    
    prg_preproc_data()
    
    Parameters
    
    Returns
    
    0 for successful execution
    
    Example
    
    prg_preproc_data()
        
    """
    
    print('~~~~~ Running base data generator ...')
    
    # generate base data
    gen_base_data(train_fpath = cons.train_data_fpath,
                  test_fpath = cons.test_data_fpath,
                  base_fpath = cons.base_data_fpath
                  )
    
    print('~~~~~ Running base data cleaner ...')
    
    # generate clean base data
    clean_base_data(base_fpath = cons.base_data_fpath,
                    base_clean_fpath = cons.base_clean_data_fpath
                    )
    
    print('~~~~~ Running feature engineer ...')
    
    # engineer new features
    feat_engineer(base_clean_2_fpath = cons.base_clean_2_data_fpath,
                  base_engin_fpath = cons.base_engin_data_fpath
                  )
    
    return 0
    
if __name__ == '__main__':
    
    prg_preproc_data()