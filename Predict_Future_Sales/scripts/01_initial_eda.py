# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 17:13:56 2020

@author: oislen
"""


import pandas as pd
import file_constants as cons
import utilities as utl

# load in the raw data
item_categories, items, sales_train, sample_submission, shops, test = utl.load_raw_files()

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
