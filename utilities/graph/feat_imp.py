# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 08:49:08 2021

@author: oislen
"""

import cons
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# set the plot, title and axis text size
plot_size_width = cons.plot_size_width
plot_size_height = cons.plot_size_height
plot_size = (plot_size_width, plot_size_height)
title_size = cons.plot_title_size
axis_text_size = cons.plot_axis_text_size
labelsize = cons.plot_label_size

def feat_imp(name, 
             model, 
             X_train,
             title = None,
             output_dir = None,
             output_fname = None
             ):
    
    """
    
    Feature Importance Plot Documentation
    
    Function Overview
    
    This function plots a bar chart of the most import features for a given model
    
    Defaults
    
    feat_imp(name, 
             model, 
             X_train,
             title = None,
             output_dir = None,
             output_fname = None
             )
    
    Parameters
    
    name - String, the model name
    model - Sklearn Model, the sklearn model to extract feature importance from
    X_train, Pandas DataFrame, the training data the model was fitted with
    title - String, the plot title, default is None
    output_dir - String, the output directory to save the plot, default is None
    output_fname - String, the output filename to save the plot as, default is None
    
    Returns
    
    0 for successful execution
    
    Example
    
    
    feat_imp(name = 'rfc', 
             model = randomforestclassifier(random_state = 123), 
             X_train = X_train,
             title = 'Random Forest Feature Importance',
             output_dir = None,
             output_fname = None
             )
    
    
    """
    
    # extract top 40 features
    indices = np.argsort(model.feature_importances_)[::-1][:40]
    
    # set the figure size
    plt.figure(figsize = plot_size)
    
    # create barplot
    g = sns.barplot(y = X_train.columns[indices][:40],
                    x = model.feature_importances_[indices][:40], 
                    orient = 'h'
                    )
    
    # if no title is given
    if title != None:
    
        # add text to the plot
        plt.title(title, fontsize = title_size)
    
    # set the x-axis label
    g.set_xlabel("Relative importance", 
                 fontsize = axis_text_size
                 )
    
    # set the y-axis label
    g.set_ylabel("Features",
                 fontsize = axis_text_size
                 )
    
    # set tick size
    g.tick_params(labelsize = labelsize)
    
    # option: save the plot
    if (output_dir != None):

        # if the output filename is given
        if output_fname != None:
            
            # set the filename
            filename = output_fname
            
        # else if the output filename is not given
        else:
            
            # create the filename
            filename = 'FeatureImportance_of_' + name + '.png'
            
        # create the filename and absolute path
        abs_path = os.path.join(output_dir, filename)
            
        # save the plot
        plt.savefig(abs_path)
    
    # show the plot
    plt.show()
    
    return 0
