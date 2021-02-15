# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 09:11:34 2021

@author: oislen
"""

#  Helper Functions to clean Titanic data and replace categorical values with numbers. 
# I can recommend the "Advanced Feature Engineering" notebook mentioned above for a deep dive
# into why and how this is done

def CleanData(data):
    clean = data.copy(deep = True)
    # Sex
    clean.drop(['Ticket', 'Name'], inplace=True, axis=1)
    clean.Sex.fillna('0', inplace=True)
    clean.loc[data.Sex != 'male', 'Sex'] = 0
    clean.loc[data.Sex == 'male', 'Sex'] = 1
    # Cabin
    clean.Cabin.fillna('0', inplace=True)
    clean.loc[data.Cabin.str[0] == 'A', 'Cabin'] = 1
    clean.loc[data.Cabin.str[0] == 'B', 'Cabin'] = 2
    clean.loc[data.Cabin.str[0] == 'C', 'Cabin'] = 3
    clean.loc[data.Cabin.str[0] == 'D', 'Cabin'] = 4
    clean.loc[data.Cabin.str[0] == 'E', 'Cabin'] = 5
    clean.loc[data.Cabin.str[0] == 'F', 'Cabin'] = 6
    clean.loc[data.Cabin.str[0] == 'G', 'Cabin'] = 7
    clean.loc[data.Cabin.str[0] == 'T', 'Cabin'] = 8
    # Embarked
    clean.loc[data.Embarked == 'C', 'Embarked'] = 1
    clean.loc[data.Embarked == 'Q', 'Embarked'] = 2
    clean.loc[data.Embarked == 'S', 'Embarked'] = 3
    clean.Embarked.fillna(0, inplace=True)
    clean.fillna(-1, inplace=True)
    return clean.astype(float)
