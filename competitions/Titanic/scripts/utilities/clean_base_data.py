# -*- coding: utf-8 -*-
"""
Created on Sat Nov  3 15:52:22 2018

@author: oislen
"""

# load in relevant libraries
import pandas as pd
import numpy as np
import cons
from utilities.clean_base_age import clean_base_age

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
   
    print('Working on pclass ...')
    
    # transfer the person's class attribute
    proc['Pclass'] = proc['Pclass'].map(cons.class_map).astype(int)

    print('Extracting title ...')

    # split name on , to extract the surname and title / firstname
    name_split = proc['Name'].str.split(pat = ',', expand = True)
    #surname = name_split[0]
    title_firstname = name_split[1]
    
    # split on . to extract title and first name
    title_split = title_firstname.str.split(pat = '.', expand = True)
    title = title_split[0].str.replace(pat = ' ', repl = '')
    #firstname = title_split[1]
    
    # map the title vlaues
    title_map = title.map(cons.title_map)

    # map the title vlaues
    status_map = title_map.map(cons.priv_map)
    
    # create title ordinal var
    proc['Title_Ord'] = status_map.map(cons.title_ord_map)
    
    # generate dummy variables for the titles
    dummy = pd.get_dummies(status_map)
    concat_objs = [proc, dummy]
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
    proc['Embarked'] = proc['Embarked'].map(cons.embarked_map).astype(int)
    
    print('Working on fare ...')

    # create fare filters
    pclass_filt = (proc['Pclass'] == 1)
    sex_filt = (proc['Sex'] == 'male')
    age_filt = (proc['Age'] > 50) & (proc['Age'] < 70)
    title_filt = (proc['Mr'] == 1)
    embarked_filt = (proc['Embarked'] == 1)
    parch_filt = (proc['Parch'] == 0)
    sibsp_filt = (proc['SibSp'] == 0)
    mean_fare_filt = pclass_filt & sex_filt & age_filt & title_filt & embarked_filt & parch_filt & sibsp_filt
    
    # calculate the mean fare for people with similar data points
    mean_fare = np.round(proc.loc[mean_fare_filt, 'Fare'].mean(skipna = True), 2)
    
    # fill in mean value
    proc['Fare'] = proc['Fare'].fillna(mean_fare)
    
    # cut up Fare
    proc['Fare_Ord'] = pd.qcut(proc['Fare'], q = 4, labels = [1, 2, 3, 4]).astype(int)
    
    if False:
        
        print('Working on cabin ...')
        
        # extract the cabin numbers
        base['Cabin_Numbers'] = base['Cabin'].str.replace(pat = '([A-Za-z ])', repl = '')
        base['Cabin_Numbers'][base['Cabin_Numbers'] == ''] = np.nan
        
        
        # extract the cabin letters
        base['Cabin_Letters'] = base['Cabin'].str.replace(pat = '([\d* ])', repl = '')
    
        base['Cabin_Letters'] = base['Cabin_Letters'].map(cons.cab_map)
        
        proc = pd.concat(objs = [proc, 
                                 pd.get_dummies(data = base['Cabin_Letters'],
                                                prefix = 'Cabin_Letters'
                                                )
                                 ],
                         axis = 1
                         )
    
        print('Working on ticket ...')
        
        # extract the ticket prefic
        ticket_prefix = base['Ticket'].str.extract(pat = '^(.*) ', expand = False)
        
        # extract the ticket number
        ticket_num = base['Ticket'].str.extract(pat = '(\d+)$', expand = False).astype(float)
        
        # fill in the empty ticket prefix with the correct value
        ticket_prefix[ticket_num.isnull()] = 'LINE'
        
        # clean the ticket prefix so that / . and spaces are removed
        ticket_prefix = ticket_prefix.str.replace(pat = '([ ./\d]*)', repl = '')
    
        proc['Ticket_Number'] = ticket_num.fillna(-1).astype(int)

    print('Filling in missing age values ...')
    
    # generate clean base age
    proc = clean_base_age(base = proc)

    print('Outputting cleaned base file ...')
    
    # subet output columns
    proc_sub = proc[cons.sub_cols]
    
    # output the dataset
    proc_sub.to_csv(base_clean_fpath,
                    sep = '|',
                    encoding = 'utf-8',
                    header = True,
                    index = False
                    )

    return 0
