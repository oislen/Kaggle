# -*- coding: utf-8 -*-
"""
Created on Sun May 16 10:55:21 2021

@author: oislen
"""

# load relevant libraries
import cons
from sub_comp_preds import sub_comp_preds
from get_comp_subs import get_comp_subs

# submit model predictions
sub_comp_preds(comp_name = cons.comp_name,
               pred_data_fpath = cons.meta_level_II_preds_fpath,
               sub_mess = "testing kaggle api"
               )

# get submission results
get_comp_subs(comp_name = cons.comp_name)
