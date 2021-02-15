# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 09:24:55 2021

@author: oislen
"""

import pandas as pd

def manualtree(df):
    # using manualtree from https://www.kaggle.com/ldfreeman3/a-data-science-framework-to-achieve-99-accuracy
    #initialize table to store predictions
    Model = pd.DataFrame(data = {'manual_tree':[]})
    #male_title = ['Master'] #survived titles

    for index, row in df.iterrows():

        #Question 1: Were you on the Titanic; majority died
        Model.loc[index, 'manual_tree'] = 0

        #Question 2: Are you female; majority survived
        if (df.loc[index, 'Sex'] == 'female'):
                  Model.loc[index, 'manual_tree'] = 1

        #Question 3A Female - Class and Question 4 Embarked gain minimum information

        #Question 5B Female - FareBin; set anything less than .5 in female node decision tree back to 0       
        if ((df.loc[index, 'Sex'] == 'female') & 
            (df.loc[index, 'Pclass'] == 3) & 
            (df.loc[index, 'Embarked'] == 'S')  &
            (df.loc[index, 'Fare'] > 8)

           ):
                  Model.loc[index, 'manual_tree'] = 0

        #Question 3B Male: Title; set anything greater than .5 to 1 for majority survived
        if ((df.loc[index, 'Sex'] == 'male') &
            (df.loc[index, 'Title'] == 3)
            ):
            Model.loc[index, 'manual_tree'] = 1
        
        
    return Model
