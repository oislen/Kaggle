# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 14:42:09 2021

@author: oislen
"""

# load relevant libraries
import cons
from sub_comp_preds import sub_comp_preds

# submit model predictions
sub_comp_preds(comp_name = cons.comp_name,
               pred_data_fpath = cons.pred_data_fpath,
               sub_mess = "testing kaggle api"
               )
