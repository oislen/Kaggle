# -*- coding: utf-8 -*-
"""
Created on Sun May 24 16:21:45 2020

@author: oislen
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import RidgeCV, Ridge, ElasticNet, Lasso, LassoCV, LassoLarsCV
from sklearn.model_selection import cross_val_score

def rmse_cv(model, X_train, y):
    rmse= np.sqrt(-cross_val_score(model, X_train, y, scoring="neg_mean_squared_error", cv = 5))
    return(rmse)

#-- Consolidate Meta Features --#

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

preds_dir = 'C:/Users/User/Documents/GitHub/Kaggle/Predict_Future_Sales/data/pred/'

for idx, f in enumerate(file):
    
    # extract out attr name
    attr_name = '_'.join(f.split('_')[0:3])
    
    preds_fpath = '{}{}'.format(preds_dir, f)
    
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
"""
meta_feat_fpath = 'C:/Users/User/Documents/GitHub/Kaggle/Predict_Future_Sales/data/model/meta_feats.feather'
join_data = pd.read_feather(meta_feat_fpath)


sns.scatterplot(x = 'gradboost_dept3_20200523', y = 'randforest_dept7_20200523', data = join_data)


#-- Prepare modelling features --#

# extract out columns
meta_cols = join_data.columns


index_cols = ['primary_key', 'ID', 'data_split', 'meta_level', 'holdout_subset_ind',
              'no_sales_hist_ind', 'year', 'month', 'date_block_num', 'item_id',
              'shop_id']

pred_cols = meta_cols[meta_cols.str.contains('_dept')].tolist()
pred_cols = ['gradboost_dept3_20200523', 'randforest_dept7_20200523']

tar_col = ['item_cnt_day']

# transform columns
trans_cols = pred_cols + tar_col

# generate histograms
for pred in pred_cols:
    sns.distplot(a = join_data[pred], bins = 100, kde = False)
    plt.show()

# apply log transformation
prep_data = join_data.copy(True)
prep_data[pred_cols] = (prep_data[pred_cols] + 1) ** (1/10000)

# normalise to mean 0 and standard deviation 1
prep_data[pred_cols] = (prep_data[pred_cols] - prep_data[pred_cols].mean()) / prep_data[pred_cols].std()

# generate histograms
for pred in pred_cols:
    sns.distplot(a = prep_data[pred], bins = 100, kde = False)
    plt.show()

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

#-- Create Level II Model --#

# ridge regression
# tune alphas
model_ridge = Ridge()
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
y_valid['meta_lvl_II_preds'] = y_valid['meta_lvl_II_preds'].apply(lambda x: 0 if x < 0 else (20 if x > 20 else x))
y_test['meta_lvl_II_preds'] = y_test['meta_lvl_II_preds'].apply(lambda x: 0 if x < 0 else (20 if x > 20 else x))
y_holdout['meta_lvl_II_preds'] = y_holdout['meta_lvl_II_preds'].apply(lambda x: 0 if x < 0 else (20 if x > 20 else x))

# plot histgrams
sns.distplot(a = y_valid['item_cnt_day'], bins = 100, kde = False)
plt.show()
sns.distplot(a = y_valid['meta_lvl_II_preds'], bins = 100, kde = False)
plt.show()
sns.distplot(a = y_valid['item_cnt_day'], bins = 100, kde = False)
plt.show()
sns.distplot(a = y_test['meta_lvl_II_preds'], bins = 100, kde = False)
plt.show()
sns.distplot(a = y_holdout['meta_lvl_II_preds'], bins = 100, kde = False)
plt.show()

# output results
submission = pd.DataFrame({"ID": y_holdout['ID'].astype(int), "item_cnt_month": y_holdout['meta_lvl_II_preds'].rename({'meta_lvl_II_preds':'item_cnt_month'})})
submission = submission.sort_values(by = ['ID']).round()

submission.to_csv('C:/Users/User/Documents/GitHub/Kaggle/Predict_Future_Sales/data/pred/stackmodel20200524.csv', index=False)

