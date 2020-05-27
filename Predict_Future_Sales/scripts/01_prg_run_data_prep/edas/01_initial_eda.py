# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 17:13:56 2020

@author: oislen
"""

# feature ideas
# is return
# is sale
# number of public holidays
# season
# month
# week
# diff in item price (sales?)
# total number of items sold by shop, month, item
# total number of items returned by shop, month, item
# total price of items sold by shop, month, item
# total price of items returned by shop, month, item
# total number of unique items by shop
# item category
# sub item category

import pandas as pd
import os
import sys 
import seaborn as sns

# add requried reference file path to paths
cwd = os.getcwd()
pwd = cwd.split('\\')
ref_dir = os.path.join('\\'.join(pwd[:-1]), 'reference')
par_dir = '\\'.join(pwd[:-2])
for path in [ref_dir, par_dir]:
    sys.path.append(path)

import file_constants as cons
import clean_utilities as utl
import numpy as np

# load in the raw data
item_categories, items, sales_train, sample_submission, shops, test = utl.load_files('raw', cons = cons)

#-- Item Categories --#

# split out item category
item_categories['item_cat'] = item_categories['item_category_name'].str.split(' - ', expand = True)[0]
item_categories['item_cat_sub'] = item_categories['item_category_name'].str.split(' - ', expand = True)[1]

# Item categories gives a description of the items category
# Has one primary key for joins

# get descriptive stats for item categories
item_categories.head()
item_categories.columns
item_categories.describe()
item_categories.dtypes
item_categories.isnull().sum()

# most common values
item_categories['item_category_id'].value_counts()
item_categories['item_category_name'].value_counts()
item_categories['item_cat'].value_counts()
item_categories['item_cat_sub'].value_counts().head(50)

item_cat = item_categories['item_cat']
item_cat = item_cat.str.replace('Payment.*', 'Payment Cards')
item_cat = item_cat.str.replace('.*[Gg]ames.*', 'Games')
item_cat.value_counts()

item_cat_sub = item_categories['item_cat_sub']
item_cat_sub = item_cat_sub.str.replace('^Audiobooks.*', 'Audiobooks')
item_cat_sub = item_cat_sub.str.replace('^CD.*', 'CD')
item_cat_sub = item_cat_sub.str.replace('^Live.*', 'Live')
item_cat_sub = item_cat_sub.str.replace('^Teaching.*', 'Teaching')
item_cat_sub.value_counts()

#-- Items --#

# Items gives a description of the items 
# Has two primary keys for joins

# get descriptive stats for items
items.head()
items.columns
items.describe()
items.dtypes
items.isnull().sum()
items['item_name'].value_counts()
items['item_id'].value_counts()
items['item_category_id'].value_counts()

#-- Sales Train --#

# Sales gives the sales information for all shops across a period of time
# has multiple foreign keys for joining all datasets

# extract date information
sales_train['year'] = sales_train['date'].str.extract('\d{2}.\d{2}.(\d{4})')
sales_train['month'] = sales_train['date'].str.extract('\d{2}.(\d{2}).\d{4}')
sales_train['day'] = sales_train['date'].str.extract('(\d{2}).\d{2}.\d{4}')
sales_train['month_year'] = sales_train['date'].str.replace('\d{2}.(\d{2}).(\d{4})', '\g<2>\g<1>')

# create shop & item combination
sales_train['shop_item_id'] = sales_train['shop_id'].astype(str) + '_' + sales_train['item_id'].astype(str)
train_shop_item = sales_train['shop_item_id'].unique()

# get descriptive stats for sales
sales_train.head()
sales_train.columns
sales_train.describe()
sales_train.dtypes
sales_train.isnull().sum()

# get value counts
sales_train['date'].value_counts(dropna = False)
sales_train['date_block_num'].value_counts(dropna = False)
sales_train['shop_id'].value_counts(dropna = False)
sales_train['item_id'].value_counts(dropna = False)
sales_train['item_price'].value_counts(dropna = False)
sales_train['item_cnt_day'].value_counts(dropna = False)
sales_train['year'].value_counts(dropna = False)
sales_train['month'].value_counts(dropna = False)
sales_train['day'].value_counts(dropna = False)

# item count day
sales_train['item_cnt_day'].min() # refunds
sales_train['item_cnt_day'].max()
(sales_train['item_cnt_day'] == 0).sum()
date_block_num_cnts = sales_train['date_block_num'].value_counts(dropna = False).sort_index()
sns.lineplot(x = date_block_num_cnts.index, y= date_block_num_cnts.values)

# date block num (year month combination)
tab = pd.crosstab(index = sales_train['month_year'], columns = sales_train['date_block_num'])
sales_train['month_year'].nunique() == sales_train['date_block_num'].nunique()

# shop id & item id combinations
sales_train['shop_id'].nunique() # 60 unique values
sales_train['item_id'].nunique() # 21807 unique values
sales_train['shop_id'].nunique() * sales_train['item_id'].nunique() # 1,308,420 item & shop combinations

# item price
sales_train['item_price'].value_counts()
sales_train['item_price'].astype(str).str.extract('\d+\.(\d{2})\d+')[0].value_counts()

#-- Shops --#

# Shops gives the names of available sjop
# has one primary key for joins

# get descriptive stats for shops
shops.head()
shops.columns
shops.describe()
shops.dtypes
shops.isnull().sum()
shops['shop_name'].value_counts()
shops['shop_id'].value_counts()

shops['shop_name'].str.extract('(".*")')[0].value_counts(dropna = False)

#-- Test --#

# Test gives the test set ids to predict for
# want to predict sales for a given item in a given shop
test.head()
test.columns
test.describe()
test.dtypes
test.isnull().sum()
test['ID'].value_counts()
test['shop_id'].value_counts()
test['item_id'].value_counts()

test['shop_id'].nunique() # 42 unique values
test['item_id'].nunique() # 5100 uniqye values
test['shop_id'].nunique() * test['item_id'].nunique() # 214299 item & shop combinations

# create shop & item combination
test['shop_item_id'] = (test['shop_id'].astype(str) + '_' + test['item_id'].astype(str)).unique()
test_shop_item = test['shop_item_id'].unique()

# check train shop item ids are in test and vica versa
train_shop_item[np.isin(train_shop_item, test_shop_item)]
test_shop_item[np.isin(test_shop_item, train_shop_item)]
