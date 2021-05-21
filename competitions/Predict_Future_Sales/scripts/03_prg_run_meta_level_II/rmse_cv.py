# -*- coding: utf-8 -*-
"""
Created on Fri May 21 13:11:43 2021

@author: oislen
"""

import numpy as np
from sklearn.model_selection import cross_val_score

def rmse_cv(model, X_train, y):
    rmse= np.sqrt(-cross_val_score(model, X_train, y, scoring="neg_mean_squared_error", cv = 5))
    return(rmse)