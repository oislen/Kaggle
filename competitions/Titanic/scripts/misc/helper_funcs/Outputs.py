# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 09:17:07 2021

@author: oislen
"""
import numpy as np


# This function rounds values to either 1 or 0, because the GeneticFunction below returns floats and no
# definite values
def Outputs(data):
    output = np.round(1.-(1./(1.+np.exp(-data))))
    return output
