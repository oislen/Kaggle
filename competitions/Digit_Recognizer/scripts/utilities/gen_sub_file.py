# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 19:25:11 2021

@author: oislen
"""

# import relevant libraries
import numpy as np
import pandas as pd
import os

# generate the submission file
def gen_sub_file(model, 
                 test_data, 
                 pred_data_fpath
                 ):
    
    """
    
    Generate Submission File Documentation
    
    Function Overview
    
    This function creates a Kaggle .csv submission file for the digit recogniser competition given a model, data and output file path.
    
    Defaults
    
    gen_sub_file(model, 
                 test_data, 
                 pred_data_fpath
                 )
    
    Parameters
    
    model - keras.model, the fitted keras model to make predictions with
    test_data - np.arry, the test data to make predictions for
    pred_data_fpath - string, the output file path to save the predictions as a .csv file.
    
    Returns
    
    0 for sucessful execution
      
    gen_sub_file(model = lenet_model, 
                 test_data = X_test, 
                 pred_data_fpath = cons.pred_data_fpath
                 )
    
    """
    
    print('checking inputs ...')
    
    # extract out the pred data fpath directory
    pred_data_dir = os.path.dirname(pred_data_fpath)
    # if os path directory does not exist
    if os.path.exists(pred_data_dir) == False:
        raise OSError('Input Error: the predictions output file path directory does not exist {}.'.format(pred_data_fpath))
    
    print('making model predictions ...')
    
    # predict results
    results = model.predict(test_data)
    
    # select the indix with the maximum probability
    results = np.argmax(results, axis = 1)
    
    print('outputting model predictions ...')
    
    # convert results into a pandas series
    results = pd.Series(results, 
                        name = "Label"
                        )
    
    # create the time id range
    time_id_range = range(1, 28001)
    
    # create timage id series
    image_id = pd.Series(time_id_range, 
                         name = "ImageId"
                         )
    
    # create a list of the concatenation objects for submission
    concat_obs = [image_id, results]
    
    # concatenate image id and results together for submission
    submission = pd.concat(objs = concat_obs,
                           axis = 1
                           )
    
    # output submission results to .csv
    submission.to_csv(pred_data_fpath,
                      index = False
                      )
    
    return 0
