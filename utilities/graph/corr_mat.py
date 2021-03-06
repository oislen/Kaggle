# -*- coding: utf-8 -*-
"""
Created on Sat Feb 13 10:21:26 2021

@author: oislen
"""

# import relevant libraries
import cons 
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib.colors as clrs

# set the plot, title and axis text size
plot_size_width = cons.plot_size_width
plot_size_height = cons.plot_size_height
plot_size = (plot_size_width, plot_size_height)
title_size = cons.plot_title_size
axis_text_size = cons.plot_axis_text_size
labelsize = cons.plot_label_size


def corr_mat(dataset,
             attrs,
             method = 'spearman',
             title = None,
             output_dir = None,
             output_fname = None
             ):
    
    """
    
    Visualise Correlation Matrix Documentation
    
    Function Overview
    
    This function generates a correlation matrix for a given dataset and list of attributes.
    The plot is created using seaborn.heatmap() and pandas.DataFrame.corr().
    Ideally the plot should be created with at max 30 variables, otherwise the plot can be very messy.
    Note, the absolute value is taken from the correlation matrix, standardising corrlation to lie on the interval [0, 1]
    Furthermore, the correlation coefficient is rounded to 2 decimal places.
    The plot can be output as a .png file to a specified directory using the path command, the plot will be named as AbsCorrelationMatrix.png    
    
    Defaults
    
    corr_mat(dataset,
             attrs,
             method = 'spearman',
             title = None,
             output_dir = None,
             output_fname = None
             )
    
    Parameters
    
    dataset - Datafrane, the dataset to generate the correlation matrix from
    attrs - List of Strings, the attribute names from the dataset to generate the correlation matrix from
    method - String, the type of correlation to perform, either 'pearson', 'spearman' or 'kendall', default is spearman
    title - String, the title of the correlation plot, default is None.
    output_dir - String, the path to output the plot to, default is None.
    output_fname - String, the filename and extension of the output plot, default is None.
    
    Returns
    
    Correlation Matrix Plot
    
    Example
    
    corr_mat(dataset = data,
             attributes = ['ccj_count_idx_12Q_pc', 'ccj_erv_1_perhse_idx_4Q_pcs', 
                           'ins_rolling_adj_count_pc', 'lr_price_delta_pcs',
                           'sil_wealth_score_pc', 'jbs_12mplus_pcd'],
             method = 'kendall',
             title = 'RI Attr Absolute Correlation Matrix',
             output_dir = '/opt/dataprojects/Analysis',
             output_fname = 'abs_kendall_corr_mat.png'
             )
    
    See Also
    
    var_corr, var_assoc
    
    References
    
    """
    
    # subset the data
    df = dataset[attrs]
    
    # calculate the correlation using the specified method
    corr = df.corr(method = method)
    
    # take the absolute value of the correlation matrix
    abs_corr = np.abs(corr)
    
    # round the correlaiton matrix to 2 decimal places
    round_corr = np.round(abs_corr, decimals = 2)
    
    # Generate a mask for the upper triangle
    mask = np.zeros_like(round_corr, dtype = np.bool)
    mask[np.triu_indices_from(mask)] = True
    
    # set the figure size
    plt.figure(figsize = plot_size)
    
    # Create the color mapper for the road safe score
    col_list = ["lightsalmon", "lightcoral", "tomato", "red", "darkred"] 
    cmap = clrs.LinearSegmentedColormap.from_list("", col_list)
    
    n_variables = len(attrs)
    
    # define the fontsize
    if n_variables <= 5:
        fontsize = 20# 80 / n_variables
    elif n_variables > 5 and n_variables <= 10:
        fontsize = 15
    elif n_variables > 10 and n_variables <= 20:
        fontsize = 10
    elif n_variables > 20 and n_variables <= 30:
        fontsize = 8
        
    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(round_corr, 
                mask = mask, 
                cmap = cmap, 
                vmax = 0.3, 
                center = 0,
                square = True, 
                linewidths = 0.5,
                annot_kws = {'fontsize':fontsize},
                cbar_kws = {"shrink":0.5},
                annot = True
                )
    
    # set the x-axis title
    plt.xlabel('Var 2', 
               fontsize = axis_text_size
               )
    
    # set the parameter ticks
    plt.tick_params(axis = 'both', 
                    labelsize = labelsize
                    )
    
    # rotate axis ticks
    plt.xticks(rotation = 45)
    plt.yticks(rotation = 'horizontal')
    
    # set the y-axis label
    plt.ylabel('Var 1', 
               fontsize = axis_text_size
               )
    
    # if no title is given
    if title != None:
    
        # add text to the plot
        plt.title(title, fontsize = title_size)
    
    # if the title is not given
    elif title == None:
        
        # create the plot title
        title = 'Absolute Correlation Matrix'
        
        # set default title
        plt.title(title, fontsize = title_size)
    
    # option: save the plot
    if (output_dir != None):
        
        # if a filename is given
        if (output_fname != None):
            
            # set the file name
            filename = output_fname
        
        # if the filename is not given
        elif (output_fname == None):
            
            # create the filename
            filename = 'AbsCorrelationMatrix.png'
            
        # create the absolute file path
        abs_path = output_dir + '/' + filename
        
        # save the plot
        plt.savefig(abs_path)
                
    # return the plot
    plt.show()