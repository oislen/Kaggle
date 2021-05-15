# -*- coding: utf-8 -*-
"""
Created on Sat May 15 08:34:59 2021

@author: oislen
"""

import seaborn as sns
import matplotlib.pyplot as plt

def plot_preds_vs_true(dataset, 
                       tar, 
                       pred, 
                       model_name, 
                       out_fpath = None
                       ):
    
     """
     
     Plot Predictions vs True Observations Documentation
     
     Function Overview
     
     This function plots the model predictions against the true observations as a scatter plot.
     
     Defaults
     
     plot_preds_vs_true(dataset, 
                       tar, 
                       pred, 
                       model_name, 
                       out_fpath = None
                       )
     
     Parameters
     
     dataset - DataFrame, the data to create the predicitons vs true observations plot from
     tar - String, the name of the target column in the given dataset
     pred - String the name of the predictions column in the given dataset
     model_name - String, the name of the model used to create the predictions
     out_fpath - String, the file path to output the plot as a .png file, default is None
     
     Returns
     
     0 for successful execution
     
     Example
     
     plot_preds_vs_true(dataset = y_valid, 
                        tar = 'item_cnt_day', 
                        pred = 'y_valid_pred', 
                        model_name = model_name, 
                        out_fpath = preds_vs_true_valid
                        )
     
     """
     
     # take a deep copy of the data
     data = dataset.copy(True)
     
     # create the initial scatter plot of predicitons vs true observations
     sns.scatterplot(x = tar, 
                     y = pred, 
                     data = data
                     )
     
     # add the model name as a title for the plot
     plt.title(model_name)
     
     # if outputting the plot
     if out_fpath != None:
         
         # write to the specified file path as a .png file
         plt.savefig(out_fpath)
         
    # print the ploe
     plt.show() 
     
     return 0