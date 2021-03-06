# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 16:40:58 2021

@author: oislen
"""

import cons
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve as lc

# set the plot, title and axis text size
plot_size_width = cons.plot_size_width
plot_size_height = cons.plot_size_height
plot_size = (plot_size_width, plot_size_height)
title_size = cons.plot_title_size
axis_text_size = cons.plot_axis_text_size
labelsize = cons.plot_label_size

def learning_curve(model,
                   X_train,
                   y_train,
                   cv = 10,
                   train_sizes = np.linspace(0.01, 1.0, 17),
                   scoring = 'accuracy',
                   title = None,
                   n_jobs = -1,
                   output_dir = None,
                   output_fname = None,
                   shuffle = False,
                   verbose = 0
                   ):
    
    
    """
    
    Visualise Learning Curve Documentation
    
    Function Overiveiw
    
    Create a function that returns learning curves for different classifiers.
    
    Defaults
    
    learning_curve(model,
                   X_train,
                   y_train,
                   scoring = 'accuracy',
                   title = None,
                   output_dir = None,
                   output_fname = None
                   #target_type = 'class'
                   )
    
    Parameters
    
    model - Sklearn CLassifier or Regressor
    X_train - Dataframe, the dataframe of predictors to train the model on.
    y_train - Dataframe, the dataframe of response to train the model on.
    scoring - String, tje scoring metric to use in the 10 fold cross validation, default is 'accuracy'.
    title - String, the title of the plot, default is None.
    output_dir - String, the path to output the plot as a .png file, default is None.
    output_fname - String, the filename and extension of the output plot, default is None.
    
    Returns
    
    Matplotlib plot of the Model Learning Curve
    
    Example
    
    vis_learning_curve(model = LogisticRegression(), 
                       X_train = X_train, 
                       y_train = y_train, 
                       scoring = 'accuracy',
                       title = 'Learning Curve for Logistic Regressor',
                       output_dir = '/opt/dataproject/Analysis',
                       output_fname = 'learning_curve_logistic.png'
                       )

    See Also
    
    metrics, tune_hyperparameters, vis_roc
    
    References
    
    https://www.kaggle.com/eraaz1/an-interpretable-guide-to-advanced-regression?scriptVersionId=7090058
    https://www.kaggle.com/eraaz1/a-comprehensive-guide-to-titanic-machine-learning
    
    """

    # set the random seed 
    seed = 123
    
    # Create feature matrix and target vector
    #X, y = X_train, y_train
    
    # random mise x and y
    #rand_idx = np.random.choice(a = range(X.shape[0]), 
    #                            size = X.shape[0], 
    #                            replace = True
    #                            )
    
    #X = X.iloc[rand_idx, :]
    #y = y.iloc[rand_idx, :]
    
    #if target_type == 'class':

    # Create CV training and test scores for various training set sizes
    train_sizes, train_scores, test_scores = lc(estimator = model, 
                                                X = X_train, 
                                                y = y_train, 
                                                cv = cv,
                                                scoring=scoring, 
                                                n_jobs = n_jobs, 
                                                train_sizes = train_sizes, 
                                                shuffle = shuffle,
                                                random_state = seed,
                                                verbose = verbose
                                                )
    # 17 different sizes of the training set

    # Create means and standard deviations of training set scores
    train_mean = np.mean(train_scores, axis = 1)
    train_std = np.std(train_scores, axis = 1)

    # Create means and standard deviations of test set scores
    test_mean = np.mean(test_scores, axis = 1)
    test_std = np.std(test_scores, axis = 1)
    
    # set the plot size
    plt.figure(figsize = plot_size)
    
    # Draw lines
    plt.plot(train_sizes, 
             train_mean, 
             'o-', 
             color = 'red',  
             label = 'Training score'
             )
    
    plt.plot(train_sizes, 
             test_mean, 
             'o-', 
             color = 'green', 
             label = 'Cross-validation score'
             )
    
    # Draw bands
    # Alpha controls band transparency.
    plt.fill_between(train_sizes, 
                     train_mean - train_std, 
                     train_mean + train_std, 
                     alpha = 0.1, 
                     color = 'r'
                     ) 
    
    plt.fill_between(train_sizes, 
                     test_mean - test_std, 
                     test_mean + test_std, 
                     alpha = 0.1, 
                     color = 'g'
                     )
    
    # if no title is given
    if title == None:
        
        # create the title of the plot
        title = 'Learning Curve'
        
        # add text to the plot
        plt.title(title, fontsize = title_size)
    
    # else if a title is given
    elif title != None:
    
        # add text to the plot
        plt.title(title, fontsize = title_size)
    
    # Create plot
    plt.xlabel('Training Set Size', fontsize = axis_text_size)
    plt.ylabel(scoring.title() + ' Score', fontsize = axis_text_size)
    
    # set the size of the axis ticks
    plt.xticks(fontsize = axis_text_size)
    plt.yticks(fontsize = axis_text_size)
    
    # plot a legend
    plt.legend(loc = 'best', fontsize  = labelsize)
    
    # set grid lines in the plot
    plt.grid()
    
    # option: save the plot
    if (output_dir != None):
        
        # if the output filename is  given
        if output_fname != None:
            
            # set the file name
            filename = output_fname
            
        # else if the output filename is not given
        elif output_fname == None:
            
            # create the filename
            filename = 'Learning_Curve.png'
            
        # create the absolute file path
        abs_path = output_dir + '/' + filename
        
        # save the plot
        plt.savefig(abs_path)
    
    # show the plot
    plt.show()