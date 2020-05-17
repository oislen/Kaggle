# -*- coding: utf-8 -*-
"""
Created on Sun May 17 12:27:58 2020

@author: oislen
"""

import pandas as pd
import seaborn as sns

def model_validation(cons):
    
    """
    """
    
    # TODO: incorporate a whole script in model evaluation here
    y_holdout['y_holdout_pred'].value_counts()
    
    # create confusion matrix
    pd.crosstab(index = y_valid['item_cnt_day'], 
                columns = y_valid['y_valid_pred']
                )
    
    # create confusion matrix
    sns.scatterplot(x = 'item_cnt_day', y = 'y_valid_pred', data = y_valid)
    
