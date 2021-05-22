# -*- coding: utf-8 -*-
"""
Created on Fri May 21 13:16:39 2021

@author: oislen
"""

import cons
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import RidgeCV, Ridge, ElasticNet, Lasso, LassoCV, LassoLarsCV
from sklearn.model_selection import cross_val_score
from consolidate_meta_features import consolidate_meta_features
from rmse_cv import rmse_cv
from plot_preds_hist import plot_preds_hist

# generate the consolidated features from the meta-level I prediction models
join_data = pd.read_feather(cons.meta_feat_fpath)

#plot scatter plot of gradboost and randforest predictions
sns.scatterplot(x = 'gradboost', 
                y = 'randforest', 
                data = join_data
                )

# extract out columns
meta_cols = join_data.columns

# generate the relevant groups of columns / attributes
tar_col = ['item_cnt_day']
index_cols = [col for col in cons.meta_level_II_base_cols if col not in tar_col]
pred_cols = [col for col in meta_cols if col not in tar_col + index_cols]

# generate histograms
for pred in pred_cols:
    sns.distplot(a = join_data[pred], bins = 100, kde = False)
    plt.show()
    
# transform columns
trans_cols = pred_cols + tar_col

# apply log transformation
prep_data = join_data.copy(True)
prep_data['gradboost'] = prep_data['gradboost'].clip(cons.lower_bound, cons.upper_bound)
prep_data[pred_cols] = np.log1p(prep_data[pred_cols])

# check for null values
if (prep_data[pred_cols].isnull().sum() > 0).any():
    raise ValueError('Null values due to log1p transformation')

# normalise to mean 0 and standard deviation 1
prep_data[pred_cols] = (prep_data[pred_cols] - prep_data[pred_cols].mean()) / prep_data[pred_cols].std()

# generate histograms
for pred in pred_cols:
    sns.distplot(a = prep_data[pred], bins = 100, kde = False)
    plt.show()
    
    
# normalise to mean 0 and standard deviation 1
prep_data[pred_cols] = (prep_data[pred_cols] - prep_data[pred_cols].mean()) / prep_data[pred_cols].std()

#-- Split Data --#

# split into train, valid, test and holdout
filt_train = prep_data['date_block_num'].isin([30, 31])
filt_valid = prep_data['date_block_num'].isin([32])
filt_test = prep_data['date_block_num'].isin([33])
filt_holdout = prep_data['date_block_num'].isin([34])

train = prep_data[filt_train]
valid = prep_data[filt_valid]
test = prep_data[filt_test]
holdout = prep_data[filt_holdout]

# split up train, valid, test and holdout into X and y
X_train = train[index_cols + pred_cols]
y_train = train[index_cols + tar_col]
X_valid = valid[index_cols + pred_cols]
y_valid = valid[index_cols + tar_col]
X_test = test[index_cols + pred_cols]
y_test = test[index_cols + tar_col]
X_holdout = holdout[index_cols + pred_cols]
y_holdout = holdout[index_cols + tar_col]

# ridge regression
# tune alphas
#model_ridge = Ridge()
model_lasso = LassoCV(alphas = [1, 0.1, 0.001, 0.0005]).fit(X_train[pred_cols], y_train['item_cnt_day'])
alphas_ridge = list(np.arange(0,20000, 500))
#alphas_lasso = list(np.arange(0.001, 1, 0.01))
cv_ridge = [rmse_cv(Ridge(alpha = alpha), X_train[pred_cols], y_train['item_cnt_day']).mean() for alpha in alphas_ridge]
#cv_lasso = [rmse_cv(Lasso(alpha = alpha), X_train[pred_cols], y_train['item_cnt_day']).mean() for alpha in alphas_lasso]
#rmse_cv(model_lasso, X_train[pred_cols], y_train['item_cnt_day']).mean()
# plot results
cv_ridge = pd.Series(cv_ridge, index = alphas_ridge)
#cv_lasso = pd.Series(cv_lasso, index = alphas_lasso)
cv_ridge.plot(title = "Validation - Just Do It")
plt.xlabel("alpha")
plt.ylabel("rmse")
cv_ridge.min()
#cv_lasso.min()
model = Ridge(alpha = cv_ridge.idxmin())
model.fit(X_train[pred_cols], y_train['item_cnt_day'])

# poisson model
pois_fam =sm.families.Poisson()
model = sm.GLM(endog = y_train['item_cnt_day'], exog = X_train[pred_cols], family = pois_fam)
model = model.fit()
model.params
model.summary()

# dtree model
model = DecisionTreeRegressor(criterion = 'mse',splitter = 'best')
model.fit(X = X_train[pred_cols], y = y_train['item_cnt_day'])

# make predictions
y_valid['meta_lvl_II_preds'] = model.predict(X_valid[pred_cols])
y_test['meta_lvl_II_preds'] = model.predict(X_test[pred_cols])
y_holdout['meta_lvl_II_preds'] = model.predict(X_holdout[pred_cols])

# clip predictions
y_valid['meta_lvl_II_preds'] = y_valid['meta_lvl_II_preds'].clip(cons.lower_bound, cons.upper_bound)
y_test['meta_lvl_II_preds'] = y_test['meta_lvl_II_preds'].clip(cons.lower_bound, cons.upper_bound)
y_holdout['meta_lvl_II_preds'] = y_holdout['meta_lvl_II_preds'].clip(cons.lower_bound, cons.upper_bound)

# plot histgrams
plot_preds_hist(dataset = y_valid, pred = 'item_cnt_day', model_name = 'poisson')
plot_preds_hist(dataset = y_valid, pred = 'meta_lvl_II_preds', model_name = 'poisson')
plot_preds_hist(dataset = y_test, pred = 'meta_lvl_II_preds', model_name = 'poisson')
plot_preds_hist(dataset = y_holdout, pred = 'meta_lvl_II_preds', model_name = 'poisson')
