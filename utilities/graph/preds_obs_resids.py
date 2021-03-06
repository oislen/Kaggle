# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 16:46:18 2021

@author: oislen
"""


import cons
import matplotlib.pyplot as plt
import seaborn as sns

# set the plot, title and axis text size
plot_size_width = cons.plot_size_width
plot_size_height = cons.plot_size_height
plot_size = (plot_size_width, plot_size_height)
title_size = cons.plot_title_size
axis_text_size = cons.plot_axis_text_size
labelsize = cons.plot_label_size


def preds_obs_resids(dataset,
                     preds = None, 
                     resids = None,
                     obs = None,
                     output_dir = None
                     ):
    
    """
    
    Visualise Predictions vs Observations Documentation
    
    Function Overview
    
    This function plots a scatterplot of two arrays with a 45 degree angle.
    Its intended use is for numeric predictions observations.
    
    Defaults
    
    vis_preds_v_obs(dataset,
                    preds = None, 
                    resids = None,
                    obs = None,
                    output_dir = None
                    )
    
    Parameters
    
    dataset - Dataframe, the data to plot the scatter plot for
    preds - String, the name of the numeric prediction column, default is None.
    resid - String, the name of numeric residual column, default is None.
    obs - String, the name of the numeric observation column, default is None.
    output_dir - String, the path to save the plots, default is None .
    
    Returns
    
    Scatter plot with 45 degree line
    
    Example
    
    vis_preds_v_obs(pred = 'pred_lungcap', 
                    obs = 'LungCap', 
                    dataset = lungcap,
                    output_dir = '/opt/dataprojects/Analysis'
                    )
    
    See Also
    
    Reference
    
    https://seaborn.pydata.org/generated/seaborn.scatterplot.html
    
    """
    
    # if predictins and observations are given
    if (preds != None) and (obs != None): 
        
        # extract the relevant series
        preds = dataset[preds]
        obs = dataset[obs]
        resids = obs - preds
    
    # else if residuals and observations are given
    elif (resids != None) and (obs != None): 
        
        # extract the relevant series
        resids = dataset[resids]
        obs = dataset[obs]
        preds = obs - resids
        
    # else if residuals and predictions are given
    elif (resids != None) and (preds != None):     
        
        # extract the relevant series
        resids = dataset[resids]
        preds = dataset[preds]
        obs = resids + preds
    
    # else if residuals, predictions and observations are given
    elif (resids != None) and (preds != None) and (obs != None):     
        
        # extract the relevant series
        resids = dataset[resids]
        preds = dataset[preds]
        obs = dataset[obs]
    
    ######################################
    #-- Prediction vs Observation Plot --#
    ######################################
    
    # Set the figure size
    plt.figure(figsize = plot_size)
    
    # calculate the ranges of x and y
    axis_min = min(min(obs), min(preds))
    axis_max = max(max(obs), max(preds))
    axis_range = axis_max - axis_min
  
    # create the scatter plot
    sns.scatterplot(x = obs, 
                    y = preds
                    )

    # add a 45 degree line
    plt.plot([axis_min - axis_range * 0.1, axis_max + axis_range * 0.1], 
             [axis_min - axis_range * 0.1, axis_max + axis_range * 0.1], 
             color = 'red')
    
    # set axis limits
    plt.xlim(axis_min - axis_range * 0.1, axis_max + axis_range * 0.1)
    plt.ylim(axis_min - axis_range * 0.1, axis_max + axis_range * 0.1)
    
    # set the title of the plot to be the plotted attribute
    plt.title('Predictions vs Observations',
              fontsize = title_size)
    
    # set the axis label and tick size
    plt.xlabel('Observed', fontsize = axis_text_size)
    plt.ylabel('Predicted', fontsize = axis_text_size)
    plt.tick_params(axis = 'both', labelsize = labelsize)
    
    # option: save the plot
    if (output_dir != None):
        
        # create the filename and absolute path
        filename = 'Residual_vs_Prediction.png'
        abs_path = output_dir + '/' + filename
        
        # save the plot
        plt.savefig(abs_path)
                
    
    # show the plot
    plt.show()
    
    ###################################
    #-- Residual vs Prediction Plot --#
    ###################################
    
    # Set the figure size
    plt.figure(figsize = plot_size)
    
    # create the scatter plot
    sns.scatterplot(x = preds, 
                    y = resids
                    )
    
    # add a hotizontal line at 0
    plt.axhline(y = 0, color = 'red')
    
    # set the title of the plot to be the plotted attribute
    plt.title('Residual vs Predictions', fontsize = title_size)
    
    # set the axis label and tick size
    plt.xlabel('Predictions', fontsize = axis_text_size)
    plt.ylabel('Residual', fontsize = axis_text_size)
    plt.tick_params(axis = 'both', labelsize = labelsize)
    
    # option: save the plot
    if (output_dir != None):
        
        # create the filename and absolute path
        filename = 'Residual_vs_Prediction.png'
        abs_path = output_dir + '/' + filename
        
        # save the plot
        plt.savefig(abs_path)
                
    
    # show the plot
    plt.show()
    
    ####################################
    #-- Residual vs Observation Plot --#
    ####################################
    
    # Set the figure size
    plt.figure(figsize = plot_size)
    
    # create the scatter plot
    sns.scatterplot(x = obs, 
                    y = resids
                    )
    
    # add a hotizontal line at 0
    plt.axhline(y = 0, color = 'red')
    
    # set the title of the plot to be the plotted attribute
    plt.title('Residual vs Observation', fontsize = title_size)
    
    # set the axis label and tick size
    plt.xlabel('Observation', fontsize = axis_text_size)
    plt.ylabel('Residual', fontsize = axis_text_size)
    plt.tick_params(axis = 'both', labelsize = labelsize)
    
    # option: save the plot
    if (output_dir != None):
        
        # create the filename and absolute path
        filename = 'Residual_vs_Observation.png'
        abs_path = output_dir + '/' + filename
        
        # save the plot
        plt.savefig(abs_path)
                
    
    # show the plot
    plt.show()
    