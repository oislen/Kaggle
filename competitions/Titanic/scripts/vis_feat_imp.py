# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 08:49:08 2021

@author: oislen
"""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def vis_feat_imp(name, 
                 model, 
                 X_train,
                 orient = 'h',
                 fontsize = 12,
                 labelsize = 9,
                 title = None
                 ):
    
    """
    """
    
    # extract top 40 features
    indices = np.argsort(model.feature_importances_)[::-1][:40]
    
    # create barplot
    g = sns.barplot(y = X_train.columns[indices][:40],
                    x = model.feature_importances_[indices][:40], 
                    orient = orient
                    )
    
    g.set_xlabel("Relative importance", 
                 fontsize = fontsize
                 )
    
    g.set_ylabel("Features",
                 fontsize = fontsize
                 )
    
    g.tick_params(labelsize = 9)
    
    if title == None:
        
        g.set_title(name + " feature importance")
