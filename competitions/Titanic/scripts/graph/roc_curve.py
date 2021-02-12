# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 16:48:19 2021

@author: oislen
"""


import cons
import matplotlib.pyplot as plt
import sklearn.metrics as metrics

# set the plot, title and axis text size
plot_size_width = cons.plot_size_width
plot_size_height = cons.plot_size_height
plot_size = (plot_size_width, plot_size_height)
title_size = cons.plot_title_size
axis_text_size = cons.plot_axis_text_size
labelsize = cons.plot_label_size


def roc_curve(obs, 
              preds,
              dataset,
              title = None,
              output_dir = None,
              output_fname = None
              ):
    
    """
    
    Visualise Reviever Operator Curve Documentation
    
    Function Overview
    
    This function plots a ROC curve for a given dataset and specifed columns.
    
    Defaults 
    
    roc_curve(obs, 
              preds,
              dataset,
              title = None,
              output_dir = None,
              output_fname = None
              )
    
    Parameters
    
    obs - String, the name of the observation column
    preds - String, the name of the prediction column
    dataset - Dataframe, the dataset to analyse
    title - String, the title of the plot, default is 'Receiver Operating Characteristic'.
    output_dir - String, the path to save the plots, default is None.
    output_fname - String, the filename and extension of the output plot, default is None.
    
    Return
    
    ROC curve
    
    Example
    
    roc_curve(obs = 'true', 
              preds = 'pred',
              dataset = test,
              title = 'ROC Curve for Test Predictions',
              output_dir = '/opt/dataprojects/Analysis',
              output_fname = 'roc_curve_test.png'
              )
    
    See Also
    
    metrics
    
    References
    
    https://en.wikipedia.org/wiki/Receiver_operating_characteristic
    
    """
     
    # extract out the series
    y_obs = dataset[obs]
    y_preds = dataset[preds]
    
    # calculate the fpr and tpr for all thresholds of the classification
    fpr, tpr, threshold = metrics.roc_curve(y_obs, y_preds)
    roc_auc = metrics.auc(fpr, tpr)
    
    # set the plot size
    plt.figure(figsize = plot_size)
    
    # plot the AUC line
    plt.plot(fpr, 
             tpr, 
             'b', 
             label = 'AUC = %0.2f' % roc_auc)
    
    # add a legend
    plt.legend(loc = 'lower right', fontsize  = labelsize)
    
    # add a red dashed 45 degree line
    plt.plot([0, 1], [0, 1],'r--')
    
    # set axis limits
    plt.xlim([0, 1])
    plt.ylim([0, 1])

    # if no title is given
    if title == None:
        
        # create the title of the plot
        title = 'Receiver Operating Characteristic'
        
        # add text to the plot
        plt.title(title, fontsize = title_size)
    
    elif title != None:
    
        # add text to the plot
        plt.title(title, fontsize = title_size)
    
    # set axis labels
    plt.xlabel('False Positive Rate', fontsize = axis_text_size)
    plt.ylabel('True Positive Rate', fontsize = axis_text_size)
    
    # set axis ticks
    plt.tick_params(axis = 'both', labelsize = labelsize)

    # add a grid to the plot
    plt.grid(color = 'lightgrey')
    
    # option: save the plot
    if (output_dir != None):
        
        # if the output filename is given
        if output_fname != None:
            
            # set the filename
            filename = output_fname
            
        # else if the output filename is not given
        elif output_fname == None:
            
            # create the filename
            filename = 'ROC_curve_' + obs + '_' + preds + '.png'
        
        # create the absolute file path
        abs_path = output_dir + '/' + filename
        
        # save the plot
        plt.savefig(abs_path)
    
    # display the plot
    plt.show()
