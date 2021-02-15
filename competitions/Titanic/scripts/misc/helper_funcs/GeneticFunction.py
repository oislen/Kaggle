# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 09:14:34 2021

@author: oislen
"""

import numpy as np

# Now may I present: The winning gen function, Inspired by Akshat's notebook:
# https://www.kaggle.com/akshat113/titanic-dataset-analysis-level-2
def GeneticFunction(data,A,B,C,D,E,F,G,H,I,J,K,L,M,N,O,P,Q,R,S,T,U,V,W,X,Y,Z,AA,AB,AC,AD,AE,AF,AG,AH,AI,AJ,AK,AL,AM):
    gen_f = ((np.minimum( ((((A + data["Sex"]) - np.cos((data["Pclass"] / AH))) * AH)),  ((B))) * AH) +
            np.maximum( ((data["SibSp"] - AC)),  ( -(np.minimum( (data["Sex"]),  (np.sin(data["Parch"]))) * data["Pclass"]))) +
            (AG * ((np.minimum( (data["Sex"]),  (((data["Parch"] / AH) / AH))) * data["Age"]) - data["Cabin"])) +
            np.minimum( ((np.sin((data["Parch"] * ((data["Fare"] - AA) * AH))) * AH)),  ((data["SibSp"] / AH))) +
            np.maximum( (np.minimum( ( -np.cos(data["Embarked"])),  (C))),  (np.sin(((data["Cabin"] - data["Fare"]) * AH)))) +
            -np.minimum( ((((data["Age"] * data["Parch"]) * data["Embarked"]) + data["Parch"])),  (np.sin(data["Pclass"]))) +
            np.minimum( (data["Sex"]),  ((np.sin( -(data["Fare"] * np.cos((data["Fare"] * W)))) / AH))) +
            np.minimum( ((O)),  (np.sin(np.minimum( (((V / AH) * np.sin(data["Fare"]))),  (D))))) +
            np.sin((np.sin(data["Cabin"]) * (np.sin((Z)) * np.maximum( (data["Age"]),  (data["Fare"]))))) +
            np.sin(((np.minimum( (data["Fare"]),  ((data["Cabin"] * data["Embarked"]))) / AH) *  -data["Fare"])) +
            np.minimum( (((AD * data["SibSp"]) * np.sin(((AJ) * np.sin(data["Cabin"]))))),  (data["Parch"])) +
            np.sin(np.sin((np.maximum( (np.minimum( (data["Age"]),  (data["Cabin"]))),  ((data["Fare"] * AK))) * data["Cabin"]))) +
            np.maximum( (np.sin(((AI) * (data["Age"] / AH)))),  (np.sin((-AF * data["Cabin"])))) +
            (np.minimum( (np.sin((((np.sin(((data["Fare"] * AH) * AH)) * AH) * AH) * AH))),  (data["SibSp"])) / AH) +
            ((data["Sex"] - data["SibSp"]) * (np.cos(((data["Embarked"] - AA) + data["Age"])) / AH)) +
            ((np.sin(data["Cabin"]) / AH) - (np.cos(np.minimum( (data["Age"]),  (data["Embarked"]))) * np.sin(data["Embarked"]))) +
            np.minimum( (AE),  ((data["Sex"] * (J * (N - np.sin((data["Age"] * AH))))))) +
            (np.minimum( (np.cos(data["Fare"])),  (np.maximum( (np.sin(data["Age"])),  (data["Parch"])))) * np.cos((data["Fare"] / AH))) +
            np.sin((data["Parch"] * np.minimum( ((data["Age"] - K)),  ((np.cos((data["Pclass"] * AH)) / AH))))) +
            (data["Parch"] * (np.sin(((data["Fare"] * (I * data["Age"])) * AH)) / AH)) +
            (D * np.cos(np.maximum( ((0.5 * data["Fare"])),  ((np.sin(N) * data["Age"]))))) +
            (np.minimum( ((data["SibSp"] / AH)),  (np.sin(((data["Pclass"] - data["Fare"]) * data["SibSp"])))) * data["SibSp"]) +
            np.tanh((data["Sex"] * np.sin((U * np.sin((data["Cabin"] * np.cos(data["Fare"]))))))) +
            (np.minimum( (data["Parch"]),  (data["Sex"])) * np.cos(np.maximum( ((np.cos(data["Parch"]) + data["Age"])),  (AM)))) +
            (np.minimum( (np.tanh(((data["Cabin"] / AH) + data["Parch"]))),  ((data["Sex"] + np.cos(data["Age"])))) / AH) +
            (np.sin((np.sin(data["Sex"]) * (np.sin((data["Age"] * data["Pclass"])) * data["Pclass"]))) / AH) +
            (data["Sex"] * (np.cos(((data["Sex"] + data["Fare"]) * ((X) * (Y)))) / AH)) +
            np.minimum( (data["Sex"]),  ((np.cos((data["Age"] * np.tanh(np.sin(np.cos(data["Fare"]))))) / AH))) +
            (np.tanh(np.tanh( -np.cos((np.maximum( (np.cos(data["Fare"])),  (L)) * data["Age"])))) / AH) +
            (np.tanh(np.cos((np.cos(data["Age"]) + (data["Age"] + np.minimum( (data["Fare"]),  (data["Age"])))))) / AH) +
            (np.tanh(np.cos((data["Age"] * ((-AH + np.sin(data["SibSp"])) + data["Fare"])))) / AH) +
            (np.minimum( (((S) - data["Fare"])),  (np.sin((np.maximum( ((AL)),  (data["Fare"])) * data["SibSp"])))) * AH) +
            np.sin(((np.maximum( (data["Embarked"]),  (data["Age"])) * AH) * (((Q) * H) * data["Age"]))) +
            np.minimum( (data["Sex"]),  (np.sin( -(np.minimum( ((data["Cabin"] / AH)),  (data["SibSp"])) * (data["Fare"] / AH))))) +
            np.sin(np.sin((data["Cabin"] * (data["Embarked"] + (np.tanh( -data["Age"]) + data["Fare"]))))) +
            (np.cos(np.cos(data["Fare"])) * (np.sin((data["Embarked"] - ((T) * data["Fare"]))) / AH)) +
            ((np.minimum( (data["SibSp"]),  (np.cos(data["Fare"]))) * np.cos(data["SibSp"])) * np.sin((data["Age"] / AH))) +
            (np.sin((np.sin((data["SibSp"] * np.cos((data["Fare"] * AH)))) + (data["Cabin"] * AH))) / AH) +
            (((data["Sex"] * data["SibSp"]) * np.sin(np.sin( -(data["Fare"] * data["Cabin"])))) * AH) +
            (np.sin((data["SibSp"] * ((((G + V) * AH) / AH) * data["Age"]))) / AH) +
            (data["Pclass"] * (np.sin(((data["Embarked"] * data["Cabin"]) * (data["Age"] - (R)))) / AH)) +
            (np.cos((((( -data["SibSp"] + data["Age"]) + data["Parch"]) * data["Embarked"]) / AH)) / AH) +
            (D * np.sin(((data["Age"] * ((data["Embarked"] * np.sin(data["Fare"])) * AH)) * AH))) +
            ((np.minimum( ((data["Age"] * A)),  (data["Sex"])) - F) * np.tanh(np.sin(data["Pclass"]))) +
            -np.minimum( ((np.cos(((AB) * ((data["Fare"] + data["Parch"]) * AH))) / AH)),  (data["Fare"])) +
            (np.minimum( (np.cos(data["Fare"])),  (data["SibSp"])) * np.minimum( (np.sin(data["Parch"])),  (np.cos((data["Embarked"] * AH))))) +
            (np.minimum( (((data["Fare"] / AH) - E)),  (C)) * np.sin((K * data["Age"]))) +
            np.minimum( ((M)),  (((np.sin(data["Fare"]) + data["Embarked"]) - np.cos((data["Age"] * (P)))))))
    
    return gen_f