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
import file_constants as cons
import utilities as utl

# load in the raw data
item_categories, items, sales_train, sample_submission, shops, test = utl.load_files('raw')

#-- Item Categories --#

# Item categories gives a description of the items category
# Has one primary key for joins

# get descriptive stats for item categories
item_categories.head()
item_categories.columns
item_categories.describe()
item_categories.dtypes
item_categories.isnull().sum()
item_categories['item_category_id'].value_counts()
item_categories['item_category_name'].value_counts()

# split out category name
item_categories['item_category_name'].str.split(' - ', expand = True)[0].value_counts()
item_categories['item_category_name'].str.split(' - ', expand = True)[1].value_counts()

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

# get descriptive stats for sales
sales_train.head()
sales_train.columns
sales_train.describe()
sales_train.dtypes
sales_train.isnull().sum()
sales_train['date'].value_counts()
sales_train['date_block_num'].value_counts()
sales_train['shop_id'].value_counts()
sales_train['item_id'].value_counts()
sales_train['item_price'].value_counts()
sales_train['item_cnt_day'].value_counts()

# investigate dates
sales_train['year'] = sales_train['date'].str.extract('\d{2}.\d{2}.(\d{4})')
sales_train['month'] = sales_train['date'].str.extract('\d{2}.(\d{2}).\d{4}')
sales_train['day'] = sales_train['date'].str.extract('(\d{2}).\d{2}.\d{4}')
sales_train['month_year'] = sales_train['date'].str.extract('\d{2}.(\d{2}.\d{4})')
sales_train['year'].value_counts(dropna = False)
sales_train['month'].value_counts(dropna = False)
sales_train['day'].value_counts(dropna = False)

# date block num (year month combination)
sales_train['date_block_num'].value_counts(dropna = False)
pd.crosstab(index = sales_train['month_year'], columns = sales_train['date_block_num'])
 
# shop id

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
