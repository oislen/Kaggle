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
                          classes,
                          normalize = False,
                          title = 'Confusion matrix',
                          cmap = plt.cm.Blues
                          ):
    
    """
    
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    
    """
    
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