# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 11:38:47 2021

@author: oislen
"""

# load relevant libraries
import numpy as np
from matplotlib import pyplot as plt
import itertools
  
def plot_confusion_matrix(conf_matrix, 
                          normalize = False,
                          title = 'Confusion matrix',
                          cmap = plt.cm.Blues
                          ):
    
    """
    
    Plot Confusion Matrix Documentation
    
    Function Overview
    
    This function prints and plots a confusion matrix as heatmap overlayed with class counts.
    Normalisation can be applied by setting `normalize = True`.
    
    Defaults
    
    plot_confusion_matrix(conf_matrix, 
                          normalize = False,
                          title = 'Confusion matrix',
                          cmap = plt.cm.Blues
                          )
    
    Parameters
    
    conf_matrix - 2D Numpy Array, the confusion matrix to plot
    normalize - Boolean, whether or not to normalise the confusion matrix by the total number of confusion matrix records, default is False
    title - String, the plot title, default is 'Confusion matrix'
    cmap - Colour Map, a matplotlib colour plot for colour settings, default is plt.cm.Blues
    
    Returns 
    
    0 for successful execution
    
    Example
    
    plot_confusion_matrix(conf_matrix = confusion_mtx, 
                          normalize = False,
                          title = 'Confusion matrix',
                          cmap = plt.cm.Blues
                          )
    
    Source 
    
    https://www.kaggle.com/yassineghouzam/introduction-to-cnn-keras-0-997-top-6
    
    """
    
    # determine the number of classes from the square confusion matrix
    num_classes = len(conf_matrix)
    
    # generate the range of classes
    classes = range(num_classes)
    
    # plot confusion matrix
    plt.imshow(conf_matrix, interpolation='nearest', cmap=cmap)
    
    # assign plot title
    plt.title(title)
    
    # assign colour bar
    plt.colorbar()
    
    # define array of class ticks
    tick_marks = np.arange(len(classes))
    
    # assign x and y ticks
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    # if normalising plot
    if normalize:
        
        # normalise confusion matrix by total records
        conf_matrix = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
    
    # set plot threshold
    thresh = conf_matrix.max() / 2.0
    
    # loop through confusion matrix rows and columns
    for i, j in itertools.product(range(conf_matrix.shape[0]), range(conf_matrix.shape[1])):
        
        # assign plot text to confusion plot 
        plt.text(x = j, 
                 y = i, 
                 s = conf_matrix[i, j],
                 horizontalalignment = "center",
                 color = "white" if conf_matrix[i, j] > thresh else "black"
                 )
    
    # set plot layout
    plt.tight_layout()
    
    # assign x and y labels
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    return 0