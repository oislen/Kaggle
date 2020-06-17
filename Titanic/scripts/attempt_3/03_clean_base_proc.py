# -*- coding: utf-8 -*-
"""
Created on Sat Nov  3 15:52:22 2018

@author: oislen
"""

"""
#####################
#-- Preliminaries --#
#####################
"""

print('Loading libraries and data ...')

# load in relevant libraries
import pandas as pd
import numpy as np

# load in data
input_dir = 'C:\\Users\\User\\Documents\\Kaggle\\Titanic\\data\\attempt_3\\'
base_name = 'base.csv'
base = pd.read_csv(input_dir + base_name, sep = '|')

# create an empty dataframe to hold the processed data
proc = pd.DataFrame()

"""
###################
#-- PassengerId --#
###################
"""

print('Working on passengerid ...')

# transfer thpassenger id attribute
proc['PassengerId'] = base['PassengerId']

"""
################
#-- Survived --#
################
"""

print('Working on survived ...')

# transfer the target survived attribute
proc['Survived'] = base['Survived']

"""
##############
#-- Pclass --#
##############
"""

print('Working on pclass ...')

# transfer the person's class attribute
proc['Pclass'] = base['Pclass'].map({1:3, 2:2, 3:1}).astype(int)

"""
############
#-- Name --#
############
"""

print('Working on name ...')

#-- Process Base --#

# split name on , to extract the surname and title / firstname
base['surname'] = base.Name.str.split(pat = ',', expand = True)[0]
title_firstname = base.Name.str.split(pat = ',', expand = True)[1]

# split on . to extract title and first name
base['title'] = title_firstname.str.split(pat = '.', expand = True)[0].str.replace(pat = ' ', repl = '')
base['firstname'] = title_firstname.str.split(pat = '.', expand = True)[1]

# the title of a person indicates the person status
status_map = {'Mr':'Mr', 
              'Miss':'Ms', 
              'Mrs':'Mrs', 
              'Master':'Master',
              'Dr':'Dr',
              'Rev':'Rev',
              'Col':'Col',
              'Major':'Major',
              'Ms':'Ms',
              'Mlle':'Ms',
              'Capt':'Capt',
              'Don':'Mr',
              'Jonkheer':'Jonkheer',
              'Sir':'Sir',
              'Lady':'Lady',
              'Mme':'Mrs',
              'Dona':'Mrs',
              'theCountess':'Countess'
              }

# map the title vlaues
base['title'] = base['title'].map(status_map)

# create a map for the title values
status_map = {'Mr':'Mr', 
              'Ms':'Ms', 
              'Mrs':'Mrs', 
              'Master':'Priv',
              'Dr':'Priv',
              'Rev':'Priv',
              'Col':'Priv',
              'Major':'Priv',
              'Capt':'Priv',
              'Jonkheer':'Priv',
              'Sir':'Priv',
              'Lady':'Priv',
              'Countess':'Priv'
              }

# map the title vlaues
base['status'] = base['title'].map(status_map)

#-- Update Proc --#

# generate dummy variables for the titles
dummy = pd.get_dummies(base['status'])
dummy = dummy.drop(columns = ['Mr'])
proc = pd.concat(objs = [proc, dummy], axis = 1)

"""
###########
#-- Sex --#
###########
"""

print('Working on sex ...')

# map the title vlaues
proc['male'] = (base['Sex'] == 'male').astype(int)

"""
###########
#-- Age --#
###########
"""

print('Working on age ...')

# transfer the age attribute
proc['Age'] = base['Age']

"""
#############
#-- SibSp --#
#############
"""

print('Working on SibSp ...')

# transfer the sibbling spouse attribute
proc['SibSp'] = base['SibSp']

"""
#############
#-- Parch --#
#############
"""

print('Working on Parch ...')

# transfer the parent child attribute
proc['Parch'] = base['Parch']

"""
##############
#-- Ticket --#
##############
"""

print('Working on ticket ...')

# extract the ticket prefic
base['Ticket_Prefix'] = base['Ticket'].str.extract(pat = '^(.*) ')

# extract the ticket number
base['Ticket_Number'] = base['Ticket'].str.extract(pat = '(\d+)$').astype(float)

# fill in the empty ticket prefix with the correct value
base['Ticket_Prefix'][base['Ticket_Number'].isnull()] = 'LINE'

# clean the ticket prefix so that / . and spaces are removed
base['Ticket_Prefix'] = base['Ticket_Prefix'].str.replace(pat = '([ ./\d]*)', repl = '')

#-- Update the Processed Dataset --#

proc['Ticket_Number'] = base['Ticket_Number'].fillna(-1).astype(int)

"""
proc = pd.concat(objs = [proc, 
                         pd.get_dummies(data = base['Ticket_Prefix'],
                                        prefix = 'Ticket_Prefix'
                                        )
                         ],
                 axis = 1)
"""

"""
############
#-- Fare --#
############
"""

print('Working on fare ...')

# calculate the mean fare for people with similar data points
mean_fare = base.Fare[(base.Pclass == 3) & (base.Sex == 'male') & (base.Age > 50) & (base.Age < 70) & (base.title == 'Mr') & (base.Embarked == 'S') & (base.Parch == 0) & (base.SibSp == 0)].mean()

proc['Fare'] = base['Fare'].fillna(mean_fare)

"""
#############
#-- Cabin --#
#############
"""

print('Working on cabin ...')


# extract the cabin numbers
base['Cabin_Numbers'] = base['Cabin'].str.replace(pat = '([A-Za-z ])', repl = '')
base['Cabin_Numbers'][base['Cabin_Numbers'] == ''] = np.nan


# extract the cabin letters
base['Cabin_Letters'] = base['Cabin'].str.replace(pat = '([\d* ])', repl = '')
cab_map = {'A':'A', 'B':'B', 'BB':'B', 'BBB':'B', 'BBBB':'B', 'C':'C', 
           'CC':'C', 'CCC':'C', 'D':'D', 'DD':'D', 'E':'E', 'EE':'E',
           'F':'F', 'FG':'F', 'FE':'F', 'G':'G', 'T':'T'
           }
base['Cabin_Letters'] = base['Cabin_Letters'].map(cab_map)

#-- Update the Processed Dataset --#

#proc['Cabin_Numbers'] = base['Cabin_Numbers'].astype(float)

proc = pd.concat(objs = [proc, 
                         pd.get_dummies(data = base['Cabin_Letters'],
                                        prefix = 'Cabin_Letters'
                                        )
                         ],
                 axis = 1
                 )


"""
################
#-- Embarked --#
################
"""

print('Working on embarked ...')

# fill the missing values with southampton 
base['Embarked'] = base['Embarked'].fillna('S')

# map the embarked locations in order of there destinations
proc['Embarked'] = base['Embarked'].map({'S':1, 'C':2, 'Q':3}).astype(int)


"""
###################
#-- Family Size --#
###################
"""

proc['FamSize'] = base['FamSize']

"""
################
#-- Is Alone --#
################
"""

proc['Alone'] = base['Alone']

"""
###############
#-- Dataset --#
###############
"""

print('Working on dataset ...')

# transfer the dataset attribute
proc['Dataset'] = base['Dataset']

"""
##############
#-- Output --#
##############
"""

print('Outputting cleaned base file ...')

# define the output location and filename
output_dir = 'C:\\Users\\User\\Documents\\Kaggle\\Titanic\\data\\attempt_3\\'
output_filename = 'base_clean.csv'

# output the dataset
proc.to_csv(output_dir + output_filename,
            sep = '|',
            encoding = 'utf-8',
            header = True,
            index = False
            )
