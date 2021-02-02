# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 11:38:47 2021

@author: oislen
"""

# load relevant libraries
from matplotlib import pyplot as plt

def plot_errors(errors_index,
                img_errors,
                pred_labels, 
                obs_labels,
                nrows = 2,
                ncols = 3
                ):
    
    """ 
    
    Plot Errors Documention
    
    Function Overview
    
    This function plots images with their predicted and real labels.
    
    Defaults
    
    plot_errors(errors_index,
                img_errors,
                pred_labels, 
                obs_labels,
                nrows = 2,
                ncols = 3
                )
    
    Parameters
    
    errors_index - np.array, the image indices to plot 
    img_errors - np.array, the image pixels 
    pred_labels - np.array, the predicted labels
    obs_labels - np.array, the observed labels
    
    Returns
    
    0 for successful execution
    
    Source 
    
    https://www.kaggle.com/yassineghouzam/introduction-to-cnn-keras-0-997-top-6
    
    """
    
    # set plot index
    idx = 0
    
    # create subplot canvas
    fig, ax = plt.subplots(nrows = nrows,
                           ncols = ncols,
                           sharex = True,
                           sharey = True
                           )
    
    # loop through rows
    for row in range(nrows):
        
        # loop through columns
        for col in range(ncols):
            
            # extract out error
            error = errors_index[idx]
            
            # plot image
            ax[row,col].imshow(img_errors[error])
            
            # extract out error info
            pred_error = pred_labels[error]
            obs_error = obs_labels[error]
            
            # create error label
            err_label = "Predicted label :{}\nTrue label :{}".format(pred_error, obs_error)
            
            # assign error label to plot
            ax[row,col].set_title(err_label)
            
            # increment plot index
            idx += 1
            
    return 0