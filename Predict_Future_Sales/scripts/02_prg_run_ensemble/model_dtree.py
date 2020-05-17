# -*- coding: utf-8 -*-
"""
Created on Sun May 17 12:20:21 2020

@author: oislen
"""

import pandas as pd
import utilities_ensemble as utl_ens
import pickle as pk
import numpy as np
from sklearn.tree import DecisionTreeRegressor

from sklearn.model_selection import GridSearchCV


model = DecisionTreeRegressor()

model_params = {'criterion':['mse'],
                'splitter':['best', 'random'],
                'max_depth':[3, 5, 7, 9],
                'max_features':['auto'],
                'random_state':[1234]
                }



# set the train, valid and test sub limits
cv_split_dict = [{'train_sub':idx, 'valid_sub':idx + 1} for idx in np.arange(start = 1, stop = 32, step = 2)]


    


