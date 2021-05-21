# -*- coding: utf-8 -*-
"""
Created on Sun May 24 16:21:45 2020

@author: oislen
"""

# load in relevant libraries
import cons
import pandas as pd
import statsmodels.api as sm
from consolidate_meta_features import consolidate_meta_features
from plot_preds_hist import plot_preds_hist

#################################
#-- Consolidate Meta Features --#
#################################

# set models to load in 
models = ['gradboost', 'randforest']

# create the input file names
preds_fnames = ['{model}_meta_lvl_II_feats.feather'.format(model = model) for model in models]

# generate the consolidated features from the meta-level I prediction models
join_data = consolidate_meta_features(preds_fnames = preds_fnames, 
                                      preds_dir = cons.pred_data_dir, 
                                      meta_feat_fpath = cons.meta_feat_fpath
                                      )


##################################
#-- Prepare modelling features --#
##################################

# extract out columns
meta_cols = join_data.columns

# generate the relevant groups of columns / attributes
tar_col = ['item_cnt_day']
index_cols = [col for col in cons.meta_level_II_base_cols if col not in tar_col]
pred_cols = [col for col in meta_cols if col not in tar_col + index_cols]

##################
#-- Split Data --#
##################

# create filter conditions to split out data
filt_train = join_data['date_block_num'].isin([30, 31])
filt_valid = join_data['date_block_num'].isin([32])
filt_test = join_data['date_block_num'].isin([33])
filt_holdout = join_data['date_block_num'].isin([34])

# split into train, valid, test and holdout
train = join_data[filt_train]
valid = join_data[filt_valid]
test = join_data[filt_test]
holdout = join_data[filt_holdout]

# split up train, valid, test and holdout into X and y
X_train = train[index_cols + pred_cols]
y_train = train[index_cols + tar_col]
X_valid = valid[index_cols + pred_cols]
y_valid = valid[index_cols + tar_col]
X_test = test[index_cols + pred_cols]
y_test = test[index_cols + tar_col]
X_holdout = holdout[index_cols + pred_cols]
y_holdout = holdout[index_cols + tar_col]

#############################
#-- Create Level II Model --#
#############################

# poisson model
pois_fam =sm.families.Poisson()
model = sm.GLM(endog = y_train['item_cnt_day'], exog = X_train[pred_cols], family = pois_fam)
model = model.fit()
model.params
model.summary()

# make predictions
y_valid['meta_lvl_II_preds'] = model.predict(X_valid[pred_cols])
y_test['meta_lvl_II_preds'] = model.predict(X_test[pred_cols])
y_holdout['meta_lvl_II_preds'] = model.predict(X_holdout[pred_cols])

# clip predictions
y_valid['meta_lvl_II_preds'] = y_valid['meta_lvl_II_preds'].apply(lambda x: 0 if x < 0 else (20 if x > 20 else x))
y_test['meta_lvl_II_preds'] = y_test['meta_lvl_II_preds'].apply(lambda x: 0 if x < 0 else (20 if x > 20 else x))
y_holdout['meta_lvl_II_preds'] = y_holdout['meta_lvl_II_preds'].apply(lambda x: 0 if x < 0 else (20 if x > 20 else x))

# plot histgrams
plot_preds_hist(dataset = y_valid, pred = 'item_cnt_day', model_name = 'poisson')
plot_preds_hist(dataset = y_valid, pred = 'meta_lvl_II_preds', model_name = 'poisson')
plot_preds_hist(dataset = y_test, pred = 'meta_lvl_II_preds', model_name = 'poisson')
plot_preds_hist(dataset = y_holdout, pred = 'meta_lvl_II_preds', model_name = 'poisson')

# create the output results dictionary
sub_dict = {"ID": y_holdout['ID'].astype(int), 
            "item_cnt_month": y_holdout['meta_lvl_II_preds'].rename({'meta_lvl_II_preds':'item_cnt_month'})
            }

# convert dictionary to dataframe
submission = pd.DataFrame(sub_dict).sort_values(by = ['ID'])

# write output predictions to a .csv file
submission.to_csv(cons.meta_level_II_preds_fpath, index = False)