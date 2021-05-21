# -*- coding: utf-8 -*-
"""
Created on Tue May 18 07:36:53 2021

@author: oislen
"""

import os
import cons
import pandas as pd

def consolidate_meta_features(preds_fnames, 
                              preds_dir, 
                              meta_feat_fpath = None
                              ):
    
    """
    
    Consoliate Meta Features Documentation
    
    Function Overview
    
    This function loads in and consolidates all meta-level II predictions for stacked modelling.
    
    Defaults
    
    consolidate_meta_features(preds_fnames, 
                              preds_dir, 
                              meta_feat_fpath = None
                              )
    
    Parameters
    
    preds_fname - List of Strings, the file names of the meta-level I predictions
    preds_dir - String, the directory path where the meta-level I predictions are stored
    meta_feat_fpath - String, the full output file path to write the consolidate features as a .feather file
    
    Returns
    
    join_data - DataFrame, the consolidated features
    
    Example
    
    consolidate_meta_features(preds_fnames = ['gradboost_meta_lvl_II_feats.feather', 'randforest_meta_lvl_II_feats.feather'], 
                              preds_dir = 'C:\\Users\\User\\Documents\\GitHub\\Kaggle\\competitions\\Predict_Future_Sales\\data\\pred', 
                              meta_feat_fpath = 'C:\\Users\\User\\Documents\\GitHub\\Kaggle\\competitions\\Predict_Future_Sales\\data\\model\\meta_feats.feather'
                              )   
    
    """
    
    print('running consolidate meta-level I features ...')
    
    # iterate over the each model type
    for idx, fname in enumerate(preds_fnames):
        
        # extract out attr name
        attr_name = fname.split('_')[0]
        
        # create the input file path for the model predictions
        preds_fpath = os.path.join(preds_dir, fname)
        
        # load in the model predictions       
        print('loading in file: {preds_fpath} ...'.format(preds_fpath = preds_fpath))
        data = pd.read_feather(preds_fpath)
        
        # if first file is being loaded
        if idx == 0:
            
            # extract all required columns for the initial file
            join_data = data[cons.meta_level_II_base_cols]
            
        # load in the model predictions
        preds_data = data[cons.meta_level_II_tar_cols].rename(columns = {'y_meta_lvl_I_pred':attr_name})
        
        print('joining predictions ...')
        
        # join the predicted data and the base data
        join_data = join_data.merge(preds_data, 
                                    on = ['primary_key'],
                                    how = 'inner'
                                    )
    
    # if output file path is not None
    if meta_feat_fpath != None:
        
        # output meta feature
        print('outputting consolidated features file: {meta_feat_fpath} ...'.format(meta_feat_fpath = meta_feat_fpath))
        join_data.to_feather(meta_feat_fpath)
        
    return join_data