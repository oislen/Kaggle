# -*- coding: utf-8 -*-
"""
Created on Sun May 24 16:21:45 2020

@author: oislen
"""

# load in relevant libraries
import cons
from consolidate_meta_features import consolidate_meta_features
from meta_level_II_model import meta_level_II_model

# set models to load in 
models = ['dtree', 'gradboost', 'randforest']

# create the input file names
preds_fnames = ['{model}_meta_lvl_II_feats.feather'.format(model = model) for model in models]

# generate the consolidated features from the meta-level I prediction models
join_data = consolidate_meta_features(preds_fnames = preds_fnames, 
                                      preds_dir = cons.pred_data_dir, 
                                      meta_feat_fpath = cons.meta_feat_fpath
                                      )

# run meta-level II model with joined features
meta_level_II_model(join_data = join_data, validation = True)
