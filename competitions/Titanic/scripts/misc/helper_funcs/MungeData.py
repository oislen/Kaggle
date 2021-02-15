# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 09:26:15 2021

@author: oislen
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from helper_funcs.manualtree import manualtree

def MungeData(data):

    title_list = [
                'Dr', 'Mr', 'Master',
                'Miss', 'Major', 'Rev',
                'Mrs', 'Ms', 'Mlle','Col',
                'Capt', 'Mme', 'Countess',
                'Don', 'Jonkheer'
                                ]

    #replacing all people's name by their titles
    def replace_names_titles(x):
        for title in title_list:
            if title in x:
                return title
    data['Title'] = data.Name.apply(replace_names_titles)
    data['Title'] = data['Title'].replace(['Miss', 'Mrs','Ms', 'Mlle', 'Lady', 'Mme', 'the Countess', 'Dona'], 'Miss/Mrs/Ms')
    data['Title'] = data['Title'].replace(['Dr', 'Col', 'Major', 'Jonkheer', 'Capt', 'Sir', 'Don', 'Rev'], 'Dr/Military/Noble/Clergy')
    data['Title'] = data.Title.map({ 'Dr':1, 'Mr':2, 'Master':3, 'Miss':4, 'Major':5, 'Rev':6, 'Mrs':7, 'Ms':8, 'Mlle':9,
                     'Col':10, 'Capt':11, 'Mme':12, 'Countess':13, 'Don': 14, 'Jonkheer':15
                    })
    data = data.drop(['Name'],axis = 1)
    data.Title.fillna(0, inplace=True)
    data['Is_Married'] = 0
    data['Is_Married'].loc[data['Title'] == 7] = 1
    # manual_tree
    data["manual_tree"] = manualtree(data)
    # Age
    data['Age'] = data.groupby(['Sex', 'Pclass'])['Age'].apply(lambda x: x.fillna(x.median()))
    # Relatives
    data['Relatives'] = data.SibSp + data.Parch
    # Fare per person
    data['Fare_per_person'] = data.Fare / np.mean(data.SibSp + data.Parch + 1)
    #data.drop(['Fare'], inplace=True, axis=1)
    med_fare = data.groupby(['Pclass', 'Parch', 'SibSp']).Fare.median()[3][0][0]
    data = data.drop(['SibSp', 'Parch'], axis=1)
    # Filling the missing value in Fare with the median Fare of 3rd class alone passenger
    data['Fare'] = data['Fare'].fillna(med_fare)
    # Ticket
    # Sex
    data.Sex.fillna('0', inplace=True)
    data.loc[data.Sex != 'male', 'Sex'] = 0
    data.loc[data.Sex == 'male', 'Sex'] = 1
    data['Ticket_Frequency'] = data.groupby('Ticket')['Ticket'].transform('count')
    data = data.drop(['Ticket'], axis=1)
    # Cabin
    data.Cabin.fillna('0', inplace=True)
    data.loc[data.Cabin.str[0] == 'A', 'Cabin'] = 1
    data.loc[data.Cabin.str[0] == 'B', 'Cabin'] = 1
    data.loc[data.Cabin.str[0] == 'C', 'Cabin'] = 1
    data.loc[data.Cabin.str[0] == 'D', 'Cabin'] = 2
    data.loc[data.Cabin.str[0] == 'E', 'Cabin'] = 2
    data.loc[data.Cabin.str[0] == 'F', 'Cabin'] = 3
    data.loc[data.Cabin.str[0] == 'G', 'Cabin'] = 3
    data.loc[data.Cabin.str[0] == 'T', 'Cabin'] = 3
    # Embarked
    data.loc[data.Embarked == 'C', 'Embarked'] = 1
    data.loc[data.Embarked == 'Q', 'Embarked'] = 2
    data.loc[data.Embarked == 'S', 'Embarked'] = 3
    data.Embarked.fillna(3, inplace=True)
    #data.fillna(0, inplace=True)
    #print(data.columns)#data["Survived"] = svd_tmp
    data["Cabin"] = data["Cabin"].astype(int)
    # now for encoding - first we scale numeric features. E.g. Fare will have bigger values as Age, which 
    # could confuse an algorithm. therefore we normalize values in the range (-1,1)
    numeric_features = ['Relatives','Fare_per_person', 'Fare', 'Age','Ticket_Frequency']
    for feature in numeric_features:  
        x = data[feature].values #returns a numpy array
        min_max_scaler = MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(x.reshape(-1, 1) )
        data[feature] = pd.DataFrame(x_scaled)

    # Categorial features
    # Now the best thing for algorithms to work with categories is to have the category values in different
    # columns as either 1 or 0. 
    cat_features = ['Pclass','Embarked', 'Sex', 'Cabin', 'Title','manual_tree','Is_Married']
    encoded_features = []
    for feature in cat_features:
        encoded_feat = OneHotEncoder().fit_transform(data[feature].values.reshape(-1, 1)).toarray()
        n = data[feature].nunique()
        cols = ['{}_{}'.format(feature, n) for n in range(1, n + 1)]
        encoded_df = pd.DataFrame(encoded_feat, columns=cols)
        encoded_df.index = data.index
        encoded_features.append(encoded_df)
    data = pd.concat([data, *encoded_features], axis=1)
    return data.astype(float)