# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 14:13:05 2021

@author: oislen
"""

# load in relevant libraries
import os
import subprocess
import cons
from get_comp_subs import get_comp_subs

def sub_comp_preds(comp_name,
                   pred_data_fpath,
                   sub_mess = "My submission message"
                   ):
    
    """
    
    Submit Competition Predictions Documentation
    
    Function Overview
    
    This functions submits competition predictions to the kaggle platform using the kaggle api.
    
    Defaults
    
    sub_comp_preds(comp_name,
                   pred_data_fpath,
                   sub_mess = "My submission message"
                   )
    
    Parameters
    
    comp_name - String, the competition to submit predictions to
    pred_data_fpath - String, the full input file path to the predictions file for submission
    sub_mess - String, the submission message, default is "My submission message"
    
    Returns
    
    0 for successful execution
    
    Example
    
    sub_comp_preds(comp_name = "digit-recognizer",
                   pred_data_fpath = "C:\\Users\\...\\preds.csv",
                   sub_mess = "My submission message"
                   )
    
    """
    
    print('checking inputs ...')
    
    # check string data types
    str_params = [comp_name, pred_data_fpath, sub_mess]
    if any([type(param) != str for param in str_params]):
        raise TypeError('Input Error: the parameters [comp_name, pred_data_fpath, sub_mess] must be string data types.')
    # check that predictions file exists
    if os.path.exists(pred_data_fpath) == False:
        raise OSError('Input Error: predictions file not found {}'.format(pred_data_fpath))
    
    print('submitting predictions to kaggle platform ...')
    
    print(pred_data_fpath)
    
    # create the base kaggle command
    kaggle_cmd_base = 'kaggle competitions submit -c {} -f {} -m'
    
    # format kaggle command with options
    kaggle_cmd = kaggle_cmd_base.format(comp_name, pred_data_fpath)
    
    # split kaggle command
    kaggle_cmd_split = kaggle_cmd.split() + [sub_mess]
    
    # execute kaggle command
    subprocess.run(kaggle_cmd_split)
    
    # get the submission results
    get_comp_subs(comp_name = comp_name)
    
    return 0