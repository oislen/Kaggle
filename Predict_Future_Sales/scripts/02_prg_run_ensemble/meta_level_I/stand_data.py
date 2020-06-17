# -*- coding: utf-8 -*-
"""
Created on Fri May 29 20:50:10 2020

@author: oislen
"""

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def stand_data():
        
    # set the input data file path
    data_fpath = 'C:/Users/User/Documents/GitHub/Kaggle/Predict_Future_Sales/data/model/model_data.feather'
    
    # load in model data
    base = pd.read_feather(data_fpath)
    
    scalar = MinMaxScaler()
    
    scalar.fit(base)
    
    scalar.transform(base)