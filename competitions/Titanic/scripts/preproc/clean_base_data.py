# -*- coding: utf-8 -*-
"""
Created on Sat Nov  3 15:52:22 2018

@author: oislen
"""

# load in relevant libraries
import pandas as pd
import numpy as np
import cons
from preproc.clean_base_age import clean_base_age

def clean_base_data(base_fpath,
                    base_clean_fpath
                    ):
    
    """
    
    Clean Base Data Documentation
    
    Function Overview
    
    This function cleans and processes the base data file.
    
    Defaults
    
    clean_base_data(base_fpath,
                    base_clean_fpath
                    )
    
    Parameters
    
    base_fpath - String, the input file path to the base dataset
    base_clean_fpath - String, the output file path to save the cleaned base dataset
    
    Returns
    
    0 for successful execution
    
    Example
    
    clean_base_data(base_fpath = 'C:\\Users\\...\\base.csv',
                    base_clean_fpath = 'C:\\Users\\...\\base_clean.csv'
                    )    
    
    """
    
    # load in data
    base = pd.read_csv(base_fpath, sep = '|')
    
    # create a copy of the base data
    proc = base.copy(deep = True)

    print('Extracting title ...')

    # extract out title
    proc['Title'] = proc['Name'].str.extract(' ([A-Za-z]+)\.', expand = False)
    
    # map the title vlaues
    proc['Title'] = proc['Title'].map(cons.title_map).map(cons.priv_map)

    # create title ordinal var
    proc['Title_Ord'] = proc['Title'].map(cons.title_ord_map)
    
    # generate dummy variables for the titles
    dummy = pd.get_dummies(proc['Title'])
    
    # create list of dummy and proc data for concatenation
    concat_objs = [proc, dummy]
    
    # concatenate data together
    proc = pd.concat(objs = concat_objs, axis = 1)
    
    print('Working on pclass ...')
    
    # generate dummy variables for the titles
    dummy = pd.get_dummies(proc['Pclass'], prefix = 'Pclass')
    
    # create list of dummy and proc data for concatenation
    concat_objs = [proc, dummy]
    
    # concatenate data together
    proc = pd.concat(objs = concat_objs, axis = 1)
    
    print('Working on sex ...')
    
    # map the title vlaues
    proc['Male'] = (proc['Sex'] == 'male').astype(int)

    print('Working on embarked ...')
    
    # calculate mode embarked 
    mode_embarked = proc['Embarked'].value_counts().index[0]
    
    # fill the missing values with southampton 
    proc['Embarked'] = proc['Embarked'].fillna(mode_embarked)
    
    # map the embarked locations in order of there destinations
    proc['Embarked_Ord'] = proc['Embarked'].map(cons.embarked_map).astype(int)
    
    print('Working on family size ...')
        
    # create a family size attribute
    proc['FamSize'] = proc['Parch'] + proc['SibSp'] + 1
    
    # determine biggest family size
    max_famsize = proc['FamSize'].max()
    
    # define the bins to cut family size into
    famsize_bins = [1, 2, 3, 5, max_famsize + 1]
    
    # define labels to cut family size into
    famsize_labels = ['Alone', 'SmallF', 'MedF', 'LargeF']
    
    # cut family size into bins
    proc['FamSize_Cat'] = pd.cut(proc['FamSize'], bins = famsize_bins, right = False, labels = famsize_labels)
    
    # generate dummy variables for the titles
    dummy = pd.get_dummies(proc['FamSize_Cat'])
    
    # create list of dummy and proc data for concatenation
    concat_objs = [proc, dummy]
    
    # concatenate data together
    proc = pd.concat(objs = concat_objs, axis = 1)
    
    print('Working on fare ...')

    # create fare filters
    pclass_filt = (proc['Pclass'] == 1)
    sex_filt = (proc['Sex'] == 'male')
    age_filt = (proc['Age'] > 50) & (proc['Age'] < 70)
    title_filt = (proc['Mr'] == 1)
    embarked_filt = (proc['Embarked'] == 'S')
    parch_filt = (proc['Parch'] == 0)
    sibsp_filt = (proc['SibSp'] == 0)
    mean_fare_filt = pclass_filt & sex_filt & age_filt & title_filt & embarked_filt & parch_filt & sibsp_filt
    
    # calculate the mean fare for people with similar data points
    mean_fare = np.round(proc.loc[mean_fare_filt, 'Fare'].mean(skipna = True), 2)
    
    # fill in mean value
    proc['Fare'] = proc['Fare'].fillna(mean_fare)
    
    # cut up Fare
    proc['Fare_Ord'] = pd.qcut(proc['Fare'], q = 4, labels = [1, 2, 3, 4]).astype(int)
    
    # log transform fare
    proc['Fare_Log'] = proc['Fare'].apply(lambda x: np.log(x + 1))
    
    print('Working on cabin ...')

    # extract the cabin letters
    proc['Cabin_Letters'] = proc['Cabin'].str.replace(pat = '([\d* ])', repl = '').map(cons.cab_map).fillna('X')

    # generate dummy variables for the titles
    dummy = pd.get_dummies(proc['Cabin_Letters'], prefix = 'Cabin')
    
    # create list of dummy and proc data for concatenation
    concat_objs = [proc, dummy]
    
    # concatenate data together
    proc = pd.concat(objs = concat_objs, axis = 1)
   
    print('Working on ticket ...')
    
    # extract the ticket prefic
    ticket_prefix = base['Ticket'].str.extract(pat = '^(.*) ', expand = False)
    
    # keep only letters and remove spaces and symbols
    ticket_prefix = ticket_prefix.str.replace(pat = '/|\.| ', repl = '')
    
    # fill in X for missing tickes prefixes
    proc['Ticket_Prefix'] = ticket_prefix.fillna('X')
    
    # generate dummy variables for the titles
    dummy = pd.get_dummies(proc['Ticket_Prefix'], prefix = 'Ticket_Prefix')
    
    # create list of dummy and proc data for concatenation
    concat_objs = [proc, dummy]
    
    # concatenate data together
    proc = pd.concat(objs = concat_objs, axis = 1)

    print('Filling in missing age values ...')
    
    # generate clean base age
    proc = clean_base_age(base = proc)
    
    # bin age into 5 categories
    proc['Age_Ord'] = pd.cut(proc['Age'], bins = 5, labels = [1, 2, 3, 4, 5]).astype(int)
    
    print('Outputting cleaned base file ...')
    
    # subet output columns
    proc_sub = proc.drop(columns = cons.drop_cols)
    
    # output the dataset
    proc_sub.to_csv(base_clean_fpath,
                    sep = '|',
                    encoding = 'utf-8',
                    header = True,
                    index = False
                    )

    return 0
