# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 16:38:46 2021

@author: oislen
"""

#####################
#-- Preliminaries --#
#####################

# import the relevant functions
import numpy as np
import pandas as pd

###############################
#-- Train Test Split Sample --#
###############################

def train_test_split_sample(dataset,
                            X,
                            y,
                            test_size = None,
                            train_size = None,
                            random_split = True,
                            sample_target = None,
                            sample_type = 'over'
                            ):
    
    """
    
    Train Test Split Sample Documentation
      
    Function Overview
    
    This function splits a dataset in to training and testing sets, either randomly or straight.
    
    Defaults
    
    train_test_split_sample(dataset,
                            X,
                            y,
                            test_size = None,
                            train_size = None,
                            random_split = True,
                            sample_target = None,
                            sample_type = 'over'
                            )
    
    Parametres
    
    dataset - Dataframe, the dataset to use.
    X - List of strings, the name of the columns to include in the X_test & X_train
    y - List of strings, the name of the columns to include in the y_test & train datasets
    test_size - Float, the proportion of data for the test dataset, must be lie between [0, 1]
    train_size - Float, the proportion of data for the training dataset, must lie between [0, 1]
    random_split - String, whether to randomly split the data into training and test sets, default is True.
    sample_target - String, the name of the target column to sample
    sample_type - String, the type of sampling to perform, either 'over' or 'under'
    
    Returns
    
    The training and test datasets as two seperate datasets
    
    Example
    
    X_test, X_train, y_test, y_train = train_test_split(dataset = Simply_Business,
                                                        y = ['Total_Premium', 'Loss_Count', 'Incurred', 'exposure',
                                                             'severity', 'frequency', 'losscost', 'lossratio',
                                                             'cancellation'],
                                                        X = data.columns.drop(['Total_Premium', 'Loss_Count', 'Incurred', 
                                                                               'exposure', 'severity', 'frequency', 
                                                                               'losscost', 'lossratio', 'cancellation']
                                                                               ),
                                                        train_size = 0.8,
                                                        test_size = 0.2,
                                                        random_split = True,
                                                        sample_target = 'cancellation',
                                                        sample_type = 'under'
                                                        )
    
    TODO:
    
    add both sampling (over and under sampling)
    
    """
    
    # take a deep copy of the dataset
    data = dataset.copy(deep = True)
    
    ######################
    #-- Error Handling --#
    ######################
    
    # check that either test_size or train_size are given
    if (test_size == None) and (train_size == None):
        
        return "Error: either test_size or train_size must be given."
    
    # check the range of test_size values
    if (test_size != None):
        
        if (test_size > 1) and (test_size < 0):
        
            return "Error: either test_size must lie in the range (0, 1)."
        
    # check the range of train_size values
    if (train_size != None):
        
        if (train_size > 1) and (train_size < 0):
            
            return "Error: either train_size must lie in the range (0, 1)."
        
    # check the combination or both train_size and test_size
    if (test_size != None) and (train_size != None):
        
        if (train_size + test_size != 1):
            
            return "Error: train_size and test_size must add to 1."
    
    # check X values
    for val in X:
    
        if val not in data.columns:
            
            return "Error: the specified X column " + val + " is not a column from the data"
    
    # check y values
    for val in y:
    
        if val not in data.columns:
            
            return "Error: the specified X column " + val + " is not a column from the data"
        
    #########################################
    #-- Split into Training and Test Sets --#
    #########################################
    
    # create analytics_rid
    data['rid'] = range(data.shape[0])
    
    #-- Random Indexes --#
    
    # if doing a random split
    if (random_split == True):
  
        # set the random seed for reproducability
        np.random.seed(1234)
    
        # if only train_size given
        if (test_size == None) and (train_size != None):
            
            # create random tranining indexes
            train_idx = np.random.choice(a = list(range(data.shape[0])), 
                                         size = int(np.round(a = (data.shape[0] * train_size), 
                                                             decimals = 0)
                                                             ),
                                         replace = False).tolist()
                                 
            # create random test indexes
            test_idx = list(set(list(range(data.shape[0]))) - set(train_idx))
            
        # if only test_size given
        elif (test_size != None) and (train_size == None):
                
            # create random testing indexes
            test_idx = np.random.choice(a = list(range(data.shape[0])), 
                                        size = int(np.round(a = (data.shape[0] * test_size), 
                                                            decimals = 0)
                                                    ),
                                        replace = False).tolist()
                                 
            # create random test indexes
            train_idx = list(set(list(range(data.shape[0]))) - set(train_idx))
        
        # if both train_size and test_size given
        elif (test_size != None) and (train_size != None):
    
            # create random tranining indexes
            train_idx = np.random.choice(a = list(range(data.shape[0])), 
                                         size = int(np.round(a = (data.shape[0] * train_size), 
                                                             decimals = 0)
                                                             ),
                                         replace = False).tolist()
                                 
            # create random test indexes
            test_idx = list(set(list(range(data.shape[0]))) - set(train_idx))
    
    #-- Straight Indexes --#
    
    # if doing a random split
    if (random_split == False):

        # if only train_size given
        if (test_size == None) and (train_size != None):
            
            # create random tranining indexes
            train_idx = list(range(int(np.round(a = (data.shape[0] * train_size), decimals = 0))))
                                 
            # create random test indexes
            test_idx = list(set(list(range(data.shape[0]))) - set(train_idx))
            
        # if only test_size given
        elif (test_size != None) and (train_size == None):
                
            # create random testing indexes
            test_idx = list(range(int(np.round(a = (data.shape[0] * test_size), decimals = 0))))
                                 
            # create random test indexes
            train_idx = list(set(list(range(data.shape[0]))) - set(train_idx))
        
        # if both train_size and test_size given
        elif (test_size != None) and (train_size != None):
    
            # create random tranining indexes
            train_idx = list(range(int(np.round(a = (data.shape[0] * train_size), decimals = 0))))
                                 
            # create random test indexes
            test_idx = list(set(list(range(data.shape[0]))) - set(train_idx))
    
    # create the training data
    train_data = data.iloc[train_idx, :]

    # create the testing data
    test_data = data.iloc[test_idx, :]

    ##################################
    #-- Sample the Target Variable --#
    ##################################
    
    # if sampling the target variable
    if sample_target != None:
    
        # reset the index of the training data
        train_data = train_data.reset_index(drop = True)
        
        # if over sampling
        if sample_type == 'over':
            
            # extract the minimum value
            mx  = train_data[sample_target].value_counts().tolist().index(train_data[sample_target].value_counts().max())
            mn  = train_data[sample_target].value_counts().tolist().index(train_data[sample_target].value_counts().min())
            
            # subset the indexes of the target variable that are equal to the minimum
            mx_indicies = list(train_data[train_data[sample_target] == mx].index)
            # subset the indexes of the target variable that are equal to the minimum
            mn_indicies = list(train_data[train_data[sample_target] == mn].index)
            
            # randomly select size from the indices
            rand_indices = np.random.choice(mn_indicies, 
                                            size = len(mx_indicies), 
                                            replace = True)
            
            # subset the data using the random indices
            sampled_data = train_data.iloc[rand_indices, :]
            
            # concatenate the sampled_data with the 
            train_data = pd.concat([train_data.iloc[mx_indicies, :], sampled_data], axis = 0)
            
        # if over sampling
        elif sample_type == 'under':
            
            # extract the maximum and minimum value
            mx  = train_data[sample_target].value_counts().tolist().index(train_data[sample_target].value_counts().max())
            mn  = train_data[sample_target].value_counts().tolist().index(train_data[sample_target].value_counts().min())
            
            # subset the indexes of the target variable that are equal to the minimum
            mx_indicies = list(train_data[train_data[sample_target] == mx].index)
            # subset the indexes of the target variable that are equal to the maxium
            mn_indicies = list(train_data[train_data[sample_target] == mn].index)
            
            # randomly select size from the indices
            rand_indices = np.random.choice(mx_indicies, 
                                            size = len(mn_indicies), 
                                            replace = True)
            
            # subset the data using the random indices
            sampled_data = train_data.iloc[rand_indices, :]
            
            # concatenate the sampled_data with the training data
            train_data = pd.concat([train_data.iloc[mn_indicies, :], sampled_data], axis = 0)
    
        # set the index of the training back to rid
        train_data = train_data.set_index('rid')
    
    ###################################
    #-- Split into X and y datasets --#
    ###################################
    
    # subset the test predictor variables
    X_test = test_data.loc[:, X]
    
    # subset the training predictor variables
    X_train = train_data.loc[:, X]
    
    # subset the training predictor varible
    y_test = test_data.loc[:, y]
    
    # subset the training reponse variable
    y_train = train_data.loc[:, y]
    
    # return both dataframes
    return X_test, X_train, y_test, y_train
    