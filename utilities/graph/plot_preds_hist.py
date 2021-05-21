# -*- coding: utf-8 -*-
"""
Created on Sat May 15 08:34:59 2021

@author: oislen
"""

import seaborn as sns
import matplotlib.pyplot as plt

def plot_preds_hist(dataset, 
                    pred, 
                    model_name, 
                    bins = 100, 
                    kde = False, 
                    out_fpath = None
                    ):
    
    """
    
    Plot Predcitions Histogram Documentation
    
    Function Overview
    
    This function plots a histogram of the model predictions
    
    Parameters
    
    plot_preds_hist(dataset, 
                    pred, 
                    model_name, 
                    bins = 100, 
                    kde = False, 
                    out_fpath = None
                    )
    
    Parameters
    
    dataset - DataFrame, the data to use for plotting the model predictions
    pred - String, the name of the predictions column in the given dataset
    model_name - String, the name of the model used to create the predictions
    bins - Integer, the number of bins to have in the histogram, defualt 100
    kde - Boolean, wheather to a kernal density estimation to the plot, default is True
    out_fpath - String, the file path to output the results as a .csv, default is None
    
    Returns
    
    0 fir successful execution
    
    Example
    
    
    plot_preds_hist(dataset = y_valid, 
                    pred = 'item_cnt_day', 
                    bins = 100, 
                    kde = False, 
                    model_name = model_name, 
                    out_fpath = true_hist_valid
                    )
    
    """
    
    # take a deep copy of the data
    data = dataset.copy(True)
    
    # create a hist of pred distribution
    sns_plot = sns.histplot(data = data[pred], 
                            bins = bins, 
                            kde = kde
                            )
    
    # add the model name to the plot as a title
    sns_plot.set(title = model_name)
    
    # if outputting the plot
    if out_fpath != None:
        
        # write to the file path as a .png
        print(out_fpath)
        sns_plot.figure.savefig(out_fpath)
        
    # print the plot
    plt.show() 
    
    return 0