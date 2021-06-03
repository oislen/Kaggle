# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 07:55:08 2021

@author: oislen
"""

import cons
import pandas as pd
import numpy as np

def gen_model_preds(test_df, 
                    model
                    ):
    
    """
    
    Generate Model Predictions Documentation
    
    Function Overview
    
    This model creates the model prediction submission file given a test set and trained RNN model.
    The results are written to disk as .csv file using the output file path defined in the cons.py module.
    
    Defaults
    
    gen_model_preds(test_df, 
                    model
                    )
    
    Parameters
    
    test_df - DataFrame, the test data to generate predictions with
    model - Keras Model, the trained RNN model
    
    Returns
    
    sub - DataFrame, the prediction submission dataframe
    
    Example
    
    gen_model_preds(test_df = test_df, 
                    model = rnn_model
                    )
    
    """
    
    # make test predictions
    # load in submission .csv file as template
    sample_submission = pd.read_csv(cons.raw_sample_submission_fpath)
    
    # apply the model to the test set and generate model predictions
    y_pre = model.predict(test_df)
    
    # reformat the predictions into a single array
    y_pre = np.round(y_pre).astype(int).reshape(3263)
    
    # create a dictionary to hold the submission info
    sub_dict = {'id':sample_submission['id'].values.tolist(),'target':y_pre}
    
    # convert the submission dictionary into a dataframe
    sub = pd.DataFrame(sub_dict)
    
    # output the submission file to disk
    sub.to_csv(cons.pred_fpath, index = False)
    
    return sub