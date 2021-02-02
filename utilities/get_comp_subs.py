# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 14:48:32 2021

@author: oislen
"""

# import relevant libraries
import subprocess

def get_comp_subs(comp_name):
    
    """
    
    Get Competition Submissions Documentation
    
    Function Overview
    
    This function returns the current submission results for a specified competition.
    
    Defaults
    
    get_comp_subs(comp_name)
    
    Parameters
    
    comp_name - String, the competition name to get previous submission results for
    
    Returns
    
    0 for successful execution
    
    Example
    
    get_comp_subs(comp_name = 'digit-recognizer')
    
    """
    
    # create the base kaggle command
    kaggle_cmd_base = 'kaggle competitions submissions -c {}'
    
    # format base kaggle command wit arguments
    kaggle_cmd = kaggle_cmd_base.format(comp_name)
    
    # split up kaggle command for running with subprocess
    kaggle_cmd_splits = kaggle_cmd.split()
    
    # run kaggle command and pipe stout
    subproc_run = subprocess.run(kaggle_cmd_splits, stdout=subprocess.PIPE)
    
    # decode the stdout for submission results
    stdout = subproc_run.stdout.decode()
    
    # print submission results
    print(stdout)
    
    return 0
