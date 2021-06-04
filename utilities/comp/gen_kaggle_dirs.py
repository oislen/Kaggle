# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 08:20:43 2021

@author: oislen
"""

# import relevant libraries
import os

def gen_kaggle_dirs():
    
    """
    
    Generate Kaggle Directories Documentation
    
    Function Overview
    
    This function generates the addotional kaggle data and report directories for the various competitions.
    
    Defaults
    
    gen_kaggle_dirs()
    
    Parameters
    
    Returns
    
    0 for successfull execution
    
    Example
    
    gen_kaggle_dirs()
    
    """
    
    # get the current working directory (kaggle utilities sub directory)
    cwd = os.getcwd()
    
    # determine the path of the parent kaggle directory
    kaggle_dir = os.path.dirname(os.path.dirname(cwd))
    
    # join on the kaggle competition sub directory to the parent kaggle directory
    kaggle_comp_dir = os.path.join(kaggle_dir, 'competitions')
    
    # define a dictionary of competitions and subdirectories to  generate
    comp_dirs = {'Digit_Recognizer':['data'],
                 'Disaster_Tweets':['data\\raw', 'data\\checkpoints', 'data\\pred', 'data\\ref'],
                 'HousePrices':['data'],
                 'Titanic':['data\\models', 'data\\preds', 'report\\arch', 'report\\model_results'],
                 'Predict_Future_Sales':['data\\raw', 'data\\clean', 'data\\base', 'data\\model', 'data\\pred', 'data\\ref',
                                         'report\\feat_imp', 'report\\cv_results', 'report\\valid_plots\\preds_hist', 'report\\valid_plots\\preds_vs_true', 'report\\valid_metrics']
                 }
             
    # iterate over the dictionary of paths to generate
    for comp, sub_dirs in comp_dirs.items():
        
        # iterate through each dubdirectory
        for sub_dir in sub_dirs:
            
            # generate the full directory path
            full_dir_path = os.path.join(kaggle_comp_dir, comp, sub_dir)
            
            # check if directory exists
            if os.path.exists(full_dir_path) == False:
                    
                # create the directory
                os.makedirs(full_dir_path)
    
    return 0

# if run script as main programe
if __name__ == '__main__':
    
    # generate the kaggle directories
    return_code = gen_kaggle_dirs()
    
    
    