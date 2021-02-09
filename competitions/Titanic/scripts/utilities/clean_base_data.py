# -*- coding: utf-8 -*-
"""
Created on Sat Nov  3 15:52:22 2018

@author: oislen
"""

# load in relevant libraries
import pandas as pd
import numpy as np
import cons

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
    
    # create an empty dataframe to hold the processed data
    proc = pd.DataFrame()
    
    print('Working on passengerid ...')
    
    # transfer thpassenger id attribute
    proc['PassengerId'] = base['PassengerId']

    print('Working on survived ...')
    
    # transfer the target survived attribute
    proc['Survived'] = base['Survived']
    
    print('Working on pclass ...')
    
    # transfer the person's class attribute
    proc['Pclass'] = base['Pclass'].map(cons.class_map).astype(int)

    print('Working on name ...')

    # split name on , to extract the surname and title / firstname
    base['surname'] = base.Name.str.split(pat = ',', expand = True)[0]
    title_firstname = base.Name.str.split(pat = ',', expand = True)[1]
    
    # split on . to extract title and first name
    base['title'] = title_firstname.str.split(pat = '.', expand = True)[0].str.replace(pat = ' ', repl = '')
    base['firstname'] = title_firstname.str.split(pat = '.', expand = True)[1]
    
    # map the title vlaues
    base['title'] = base['title'].map(cons.title_map)

    # map the title vlaues
    base['status'] = base['title'].map(cons.priv_map)
    
    # generate dummy variables for the titles
    dummy = pd.get_dummies(base['status'])
    dummy = dummy.drop(columns = ['Mr'])
    proc = pd.concat(objs = [proc, dummy], axis = 1)
    
    print('Working on sex ...')
    
    # map the title vlaues
    proc['male'] = (base['Sex'] == 'male').astype(int)

    print('Working on age ...')
    
    # transfer the age attribute
    proc['Age'] = base['Age']

    print('Working on SibSp ...')
    
    # transfer the sibbling spouse attribute
    proc['SibSp'] = base['SibSp']

    print('Working on Parch ...')
    
    # transfer the parent child attribute
    proc['Parch'] = base['Parch']

    print('Working on ticket ...')
    
    # extract the ticket prefic
    base['Ticket_Prefix'] = base['Ticket'].str.extract(pat = '^(.*) ')
    
    # extract the ticket number
    base['Ticket_Number'] = base['Ticket'].str.extract(pat = '(\d+)$').astype(float)
    
    # fill in the empty ticket prefix with the correct value
    base['Ticket_Prefix'][base['Ticket_Number'].isnull()] = 'LINE'
    
    # clean the ticket prefix so that / . and spaces are removed
    base['Ticket_Prefix'] = base['Ticket_Prefix'].str.replace(pat = '([ ./\d]*)', repl = '')

    proc['Ticket_Number'] = base['Ticket_Number'].fillna(-1).astype(int)

    print('Working on fare ...')
    
    # calculate the mean fare for people with similar data points
    mean_fare = base.Fare[(base.Pclass == 3) & (base.Sex == 'male') & (base.Age > 50) & (base.Age < 70) & (base.title == 'Mr') & (base.Embarked == 'S') & (base.Parch == 0) & (base.SibSp == 0)].mean()
    
    proc['Fare'] = base['Fare'].fillna(mean_fare)

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

    print('Working on embarked ...')
    
    # fill the missing values with southampton 
    base['Embarked'] = base['Embarked'].fillna('S')
    
    # map the embarked locations in order of there destinations
    proc['Embarked'] = base['Embarked'].map(cons.embarked_map).astype(int)
    
    proc['FamSize'] = base['FamSize']

    proc['Alone'] = base['Alone']

    print('Working on dataset ...')
    
    # transfer the dataset attribute
    proc['Dataset'] = base['Dataset']

    print('Outputting cleaned base file ...')
    
    # output the dataset
    proc.to_csv(base_clean_fpath,
                sep = '|',
                encoding = 'utf-8',
                header = True,
                index = False
                )

    return 0

if __name__ == '__main__':
    
    clean_base_data(cons.base_data_fpath,
                    cons.base_clean_data_fpath
                    )