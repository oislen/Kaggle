# -*- coding: utf-8 -*-
"""
Created on Tue May 18 07:36:53 2021

@author: oislen
"""

import cons

def consolidate_meta_features():
    
    """
    """
    
    file = ['dtree_dept3_20200523_meta_lvl_II_feats.feather',
            'dtree_dept5_20200523_meta_lvl_II_feats.feather',
            'dtree_dept7_20200523_meta_lvl_II_feats.feather',
            'gradboost_dept3_20200523_meta_lvl_II_feats.feather',
            'gradboost_dept5_20200523_meta_lvl_II_feats.feather',
            'gradboost_dept7_20200523_meta_lvl_II_feats.feather',
            'randforest_dept3_20200523_meta_lvl_II_feats.feather',
            'randforest_dept5_20200523_meta_lvl_II_feats.feather',
            'randforest_dept7_20200523_meta_lvl_II_feats.feather'
            ]
 
    for idx, f in enumerate(file):
        
        # extract out attr name
        attr_name = '_'.join(f.split('_')[0:3])
        
        preds_fpath = '{}{}'.format(cons.preds_dir, f)
        
        data = pd.read_feather(preds_fpath)
        
        if idx == 0:
            
            base_cols = ['primary_key', 'ID', 'data_split', 'meta_level', 'holdout_subset_ind',
                         'no_sales_hist_ind', 'year', 'month', 'date_block_num', 'item_id',
                         'shop_id', 'item_cnt_day']
            
            join_data = data[base_cols]
            
        pred_cols = ['primary_key', 'y_meta_lvl_I_pred']
        
        preds_data = data[pred_cols].rename(columns = {'y_meta_lvl_I_pred':attr_name})
        
        join_data = join_data.merge(preds_data, on = ['primary_key'], how = 'inner')
    
    # output meta feature
    meta_feat_fpath = 'C:/Users/User/Documents/GitHub/Kaggle/Predict_Future_Sales/data/model/meta_feats.feather'
    join_data.to_feather(meta_feat_fpath)