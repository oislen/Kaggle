# -*- coding: utf-8 -*-
"""
Created on Sun Feb 14 11:42:30 2021

@author: oislen
"""

import os
import sys
scripts_dir = os.path.dirname(os.getcwd())
sys.path.append(scripts_dir)

import cons
import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score

from helper_funcs.GP_deap import GP_deap
from helper_funcs.MungeData import MungeData
from helper_funcs.CleanData import CleanData
from helper_funcs.GeneticFunction import GeneticFunction
from helper_funcs.Outputs import Outputs
from helper_funcs import Coef 

# load data for your genetic algorithm
train = pd.read_csv(cons.train_data_fpath)
test = pd.read_csv(cons.test_data_fpath)

#-- Training GP --#

pass_id_train = train["PassengerId"] # copy it before deleting it in the MungeData step
survived_train = train["Survived"] # copy it before deleting it in the MungeData step
pass_id_test = test["PassengerId"] # copy it before deleting it in the MungeData step

evolved_train = MungeData(train)
evolved_test = MungeData(test)

# Starts the genetic function. Remember this can have a huge comp. effort depending on the value you set 
# above in HOWMANYITERS
GeneticFunctionObject = GP_deap(evolved_train)
# optional, save our genetic function for later, good idea if computation took ages
with open(cons.gf_fpath,"wb") as file:
    pickle.dump(GeneticFunction,file)

# drop PassengerId because we will not use it
train_nparray = evolved_train.drop(columns = ["PassengerId","Survived"]).values.tolist() 

trainPredictions = Outputs(np.array([GeneticFunctionObject(*x) for x in train_nparray]))
print("Your gp fitting score based on training set (Remember, Kaggle/Test set score will be different):")
print(accuracy_score(survived_train.astype(int),trainPredictions.astype(int)))
pd_train = pd.DataFrame({'PassengerId': pass_id_train.astype(int),
                        'Predicted': trainPredictions.astype(int),
                        'Survived': survived_train.astype(int)})

# Test set submission
# drop PassengerId because we will not use it
evoled_test = evolved_test.drop(["PassengerId"],axis=1) 
test_nparray = evolved_test.values.tolist()
testPredictions = Outputs(np.array([GeneticFunctionObject(*x) for x in test_nparray]))
pd_test = pd.DataFrame({'PassengerId': pass_id_test.astype(int),
                        'Survived': testPredictions.astype(int)})
pd_test['Survived'].value_counts()

#-- PreFitted Model --#

# prep data
cleanedTrain = CleanData(train)
cleanedTest = CleanData(test)

# run a check on the Training dataset. See section "Programm your own gen. algorithm" below on how to 
# construct your own genetic algorithm
thisArray = Coef.BIG.copy()
# make training predictions
trainPredictions = Outputs(GeneticFunction(cleanedTrain,thisArray[0],thisArray[1],thisArray[2],thisArray[3],thisArray[4],thisArray[5],thisArray[6],thisArray[7],thisArray[8],thisArray[9],thisArray[10],thisArray[11],thisArray[12],thisArray[13],thisArray[14],thisArray[15],thisArray[16],thisArray[17],thisArray[18],thisArray[19],thisArray[20],thisArray[21],thisArray[22],thisArray[23],thisArray[24],thisArray[25],thisArray[26],thisArray[27],thisArray[28],thisArray[29],thisArray[30],thisArray[31],thisArray[32],thisArray[33],thisArray[34],thisArray[35],thisArray[36],thisArray[37],thisArray[38]))
pdcheck = pd.DataFrame({'Survived': trainPredictions.astype(int)})
ret = pdcheck.Survived.where(pdcheck["Survived"].values==cleanedTrain["Survived"].values).notna()
t,f = ret.value_counts()
score = 100/(t+f)*t
print("Training set score for prefitted gp model: ",score)
# create training frame
pdtrain = pd.DataFrame({'PassengerId': cleanedTrain.PassengerId.astype(int),
                        'Predicted': trainPredictions.astype(int),
                        'Survived': cleanedTrain.Survived.astype(int)})
# check results
pd.crosstab(index = pdtrain['Predicted'], columns = pdtrain['Survived'])

# generate predictions
testPredictions = Outputs(GeneticFunction(cleanedTest,thisArray[0],thisArray[1],thisArray[2],thisArray[3],thisArray[4],thisArray[5],thisArray[6],thisArray[7],thisArray[8],thisArray[9],thisArray[10],thisArray[11],thisArray[12],thisArray[13],thisArray[14],thisArray[15],thisArray[16],thisArray[17],thisArray[18],thisArray[19],thisArray[20],thisArray[21],thisArray[22],thisArray[23],thisArray[24],thisArray[25],thisArray[26],thisArray[27],thisArray[28],thisArray[29],thisArray[30],thisArray[31],thisArray[32],thisArray[33],thisArray[34],thisArray[35],thisArray[36],thisArray[37],thisArray[38]))
# make submission file
pdtest = pd.DataFrame({'PassengerId': cleanedTest.PassengerId.astype(int),
                        'Survived': testPredictions.astype(int)})
# write submission file
pdtest.to_csv(cons.pred_data_fpath.format('gpc'), index=False)
