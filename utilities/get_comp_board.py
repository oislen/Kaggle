# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 19:41:04 2021

@author: oislen
"""

# import relevant libraries
import subprocess

def get_comp_board(comp_name):
    
    """
    
    Get Competition Board Documentation
    
    Function Overview
    
    This function returns the competition leader board for a specified competition.
    
    Defaults
    
    get_comp_board(comp_name)
    
    Parameters
    
    comp_name - String, the competition name to get the leader board for
    
    Returns
    
    0 for successful execution
    
    Example
    
    get_comp_board(comp_name = 'digit-recognizer')
    
    """
    
    # create the base kaggle command
    kaggle_cmd_base = 'kaggle competitions leaderboard -c {} -s '
    
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
