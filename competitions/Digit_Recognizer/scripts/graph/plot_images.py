# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 11:37:11 2021

@author: oislen
"""

# load relevant libraries
import numpy as np
from matplotlib import pyplot as plt

def plot_images(data, 
                n_cols = 5, 
                n_rows = 1
                ):
    
    """
    
    Plot Images Documentation
    
    Function Overview
    
    This function plots raw images for a given numpy array of image pixels.
    The function outputs n_cols * n_rows images.
    
    Defaults
    
    plot_images(data, 
                n_cols = 5, 
                n_rows = 1
                )
    
    Parameters
    
    data - Numpy array, the image pixels to plot
    n_cols - Integer, the number of columnar images to plot, default is 5
    n_rows - Integer, the number of row images to plot, default is 1
    
    Returns
    
    0 for successful execution
    
    Example
    
    plot_images(data = X_train, 
                n_cols = 5, 
                n_rows = 1
                )
    
    """
    
    # determine number of images to plot
    n_images = n_cols * n_rows
    
    # set figure size dynamically based on number of rows and columns
    plt.figure(figsize = (3 * n_cols, 3 * n_rows))
    
    # generate some random image indices
    rand_image_idxs = np.random.randint(len(data), size = n_images)
    
    # enumerate through the random image indicies
    for idx, val in enumerate(rand_image_idxs):
        
        # generate subplot
        plt.subplot(n_rows, n_cols, idx + 1)
        
        # remove axis from image
        plt.axis('off')
        
        # plot image
        plt.imshow(data[val])
    
    # show plot
    plt.show()
    
    return 0