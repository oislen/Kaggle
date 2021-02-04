# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 17:07:52 2021

@author: oislen
"""

import matplotlib.pyplot as plt
import numpy as np

def plot_heatmap(images, 
                 image_preds, 
                 n_cols = 5, 
                 n_rows = 1
                 ):
    
    """
    
    Plot Heatmap Documentation
    
    Function Overview
    
    This function create a comparison heatmap of image predictions with the original image
    
    Defaults
    
    plot_heatmap(images, 
                 image_preds, 
                 n_cols = 5, 
                 n_rows = 1
                 )
    
    Parameters
    
    images - np.array, the original input images 
    image_preds - np.array, the model predictions for the orginal input image
    n_cols - integer, the number of columnar images to plot, default is 5
    n_rows - integer, the number of row images to plot, default is 1
    
    Returns
    
    0 for successful execution
    
    Example
    
    plot_heatmap(images = X_valid, 
                 image_preds = fcnn_model.predict(X_valid)[:, :, :, 1], 
                 n_cols = 5, 
                 n_rows = 1
                 )
    
    """
    
    # set the plot size
    plt.figure(figsize = (3 * n_cols, 2 * 3 * n_rows))
    
    # loop through each subplot
    for n,i in enumerate(np.arange(n_cols * n_rows)):
        
        # create the above subplot
        plt.subplot(2 * n_rows, n_cols, n + 1)
        
        # turn off axises
        plt.axis('off')
        
        # plot the image in the above subplot
        plt.imshow(images[i])

        # create the below subplot
        plt.subplot(2 * n_rows, n_cols, n + 1 + n_cols)
        
        # turn off the axis
        plt.axis('off')
        
        # plot the image predictions in the below subplot
        plt.imshow(image_preds[i])
        
    # show the plot
    plt.show()
    
    return 0