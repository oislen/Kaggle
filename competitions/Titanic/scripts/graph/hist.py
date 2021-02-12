# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 16:29:15 2021

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

def hist(dataset, 
         num_var,
         bins = None,
         hist = True,
         kde = False,
         color  = 'royalblue',
         norm_hist = False,
         cumulative = False,
         bound = None,
         title = None,
         xlab = None,
         vline = None,
         output_dir = None,
         output_fname = None
         ):
    
    """
    
    Visualise Histogram Documentation
    
    Function Overview
    
    Plots a Histogram / Kernel Desity Plot for a set of numeric varibales in a specified dataset
    The function uses the seaborn function distplot().
    
    Defaults
    
    hist(dataset, 
         num_var,
         bins = None,
         hist = True,
         kde = False,
         color  = 'royalblue',
         norm_hist = False,
         cumulative = False,
         bound = None,
         title = None,
         xlab = None,
         vline = None,
         output_dir = None,
         output_fname = None
         )
    
    Parameters
    
    dataset - Dataframe, the dataset
    num_var - List of strings, the name of the numeric variables to plot along the x-axis.
    bins - Integer, the number of bins to plot in the histogram.
    hist - Boolean, whether to plot a standard histogram, default is True.
    kde - Boolean, whether to plot a kernel density estimatied curve, default is False.
    color - String, the color to plot for the histogram, default is 'royalblue'.
    norm_hist - Boolean, whether to normalise the histogram, default is False.
    bound - List of floats, a bound to create a histogram for, e.g. [1, 3], default is None.
    title - String, the title of the plot.
    xlab - String, label for the x-axis
    vline - Float, the point to plot a vertical line through the x-axis
    output_dir - String, the path to save the plots, default is None.
    output_fname - String, the filename and extension of the output plot, default is None.
    
    Returns
    
    UA seaborn histogram / density plot
    
    Example
    
    hist(dataset = lungcap,
         num_var = ['Age'],
         bins = 40,
         hist = True,
         kde = False,
         norm_hist = False,
         bound = [10, 15],
         title = 'Histogram of Ages 10 - 15',
         xlab = 'Age Variable',
         output_dir = '/opt/dataprojects/Analysis',
         output_fname = 'histogram_age.png'
         )
    
    See Also
    
    visualise.count, visualise.boxplot, visualise.violin, visualise.scatter, visualise.regplot
    
    References        
    
    https://seaborn.pydata.org/generated/seaborn.distplot.html
    
    """
    
    #-- Error Handling --#
    
    if (kde == False) and (hist == False):
        
        print("Input Error: kde == False and hist == False, thus nothing to plot.")
    
        return 
    
    # check the attribute column is in the dataset
    for num in num_var:
        
        # if the attribute column is not in the dataset
        if num not in dataset.columns:
            
            # return this error message
            print("Input Error: The specified numeric attribute column " + num + " is not a column from the dataset")
            
            return
    
    #-- Create Histogram --#
    
    # determine which of the numeric columns are categories and which are numeric
    num_data_types = dataset[num_var].dtypes
    int_vars = num_data_types[num_data_types == 'int64'].index.tolist()
    float_vars = num_data_types[num_data_types == 'float64'].index.tolist()
    num_var = int_vars + float_vars
    
    # if bounding the data
    if bound != None:
        
        # extract lower and upper bound
        lb = min(bound)
        ub = max(bound)
    
    # use a for loop to iterate through the dataset
    for num in num_var :
        
        # remove NaN values
        series = dataset[num][dataset[num].notnull()]
        
        # if bounding the data
        if bound != None:
            
            # filter data according to bounds
            series = series[(series >= lb) & (series <= ub)]
        
        # set the figure size
        plt.figure(figsize = plot_size)
        
        # create the histogram
        sns.distplot(a = series,
                     bins = bins,
                     hist = hist,
                     kde = kde,
                     color = color,
                     norm_hist = norm_hist,
                     vertical  = False,
                     hist_kws = dict(cumulative = cumulative),
                     kde_kws = dict(cumulative = cumulative)
                     )
        
        # if no title is given
        if title != None:
        
            # add text to the plot
            plt.title(title, fontsize = title_size)
        
        # else if a title is given
        elif title == None:
            
            # if plotting both a histogram and density
            if (kde == True) and (hist == True):
            
                # create a title for the plot
                plot_title = 'Histogram / Density Plot of ' + num.title()
            
            # else if plotting only a histogram
            elif (kde == False) and (hist == True):
            
                # create a title for the plot
                plot_title = 'Histogram of ' + num.title()
            
            # else if plotting only a Density Plot
            elif (kde == True) and (hist == False):
            
                # create a title for the plot
                plot_title = 'Density Plot of ' + num.title()
                
            # add text to the plot
            plt.title(plot_title, fontsize = title_size)
        
        # xlabel is given
        if xlab != None:
            plt.xlabel(xlabel = xlab, fontsize = axis_text_size)
        else:
            # set the axis sizes
            plt.xlabel(num, fontsize = axis_text_size)
        
        plt.ylabel('Frequency', fontsize = axis_text_size)
        
        # set the axis tick sizes
        plt.tick_params(axis = 'both', labelsize = labelsize)
        
        # add a grid to the plot
        plt.grid(color = 'lightgrey')
        
        # if setting a horizontal line
        if vline != None:
        
            # add the horizontal line
            plt.axvline(x = vline,
                        color = 'red',
                        linestyle = '--'
                        )
    
        
        # option: save the plot
        if (output_dir != None):
            
            # if plotting both a histogram and density
            if (kde == True) and (hist == True):
                
                # if the output filename is given
                if output_fname != None:
                    
                    # set the filename
                    filename = output_fname
                    
                # else if the output filename is not given
                elif output_fname == None:
                    
                    # create the filename
                    filename = 'HistogramDensity_of_' + num + '.png'

            
            # else if plotting only a histogram
            elif (kde == False) and (hist == True):
            
                # if the output filename is given
                if output_fname != None:
                    
                    # set the filename
                    filename = output_fname
                    
                # else if the output filename is not given
                elif output_fname == None:
                    
                    # create the filename
                    filename = 'Histogram_of_' + num + '.png'
            
            # else if plotting only a Density Plot
            elif (kde == True) and (hist == False):
            
                # if the output filename is given
                if output_fname != None:
                    
                    # set the filename
                    filename = output_fname
                    
                # else if the output filename is not given
                elif output_fname == None:
                    
                    # create the filename
                    filename = 'Density_of_' + num + '.png'
                
            # create the filename and absolute path
            abs_path = output_dir + '/' + filename
                
            # save the plot
            plt.savefig(abs_path)
        
        # show the plot
        plt.show()
   