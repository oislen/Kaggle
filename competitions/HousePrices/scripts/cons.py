# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 10:42:07 2021

@author: oislen
"""

# load libraries
import os
import sys

# set programme constants
comp_name = 'house-prices-advanced-regression-techniques'
download_data = True
unzip_data = True
del_zip = True

# set .csv constants
sep = ','
encoding = 'utf-8'
header = True
index = False

#-- File Path Constants --#

# set directories
git_dir = 'C:\\Users\\User\\Documents\\GitHub'
root_dir = os.path.join(git_dir, 'Kaggle')
va_dir = os.path.join(git_dir, 'value_analysis')
utilities_dir = os.path.join(root_dir, 'utilities')
houseprices_comp_dir = os.path.join(root_dir, 'competitions\\HousePrices')
scripts_dir = os.path.join(houseprices_comp_dir, 'scripts')
data_dir = os.path.join(houseprices_comp_dir, 'data')
report_dir = os.path.join(houseprices_comp_dir, 'report')

# create file names
train_data_fname = 'train.csv'
test_data_fname = 'test.csv'
base_data_fname = 'base.csv'
clean_data_fname = 'clean.csv'
glm_feat_imp_fname = 'GLM_feat_imp.csv'

# create file paths
train_data_fpath = os.path.join(data_dir, train_data_fname)
test_data_fpath = os.path.join(data_dir, test_data_fname)
base_data_fpath = os.path.join(data_dir, base_data_fname)
clean_data_fpath = os.path.join(data_dir, clean_data_fname)
glm_feat_imp_fpath = os.path.join(data_dir, glm_feat_imp_fname)

# append utilities directory to path
for p in [utilities_dir, va_dir]:
    sys.path.append(p)

#-- Cleaning Constants --#
    
tar_cols = ['Id', 'Dataset', 'SalePrice', 'logSalePrice']

# columns to fill in zeros for
zero_imp = ['MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath', 'GarageCars', 'GarageArea']

# columns to fill in medians for
median_imp = ['LotFrontage']

# columns to impute mode for
mode_imp = ['MSZoning', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'BsmtFinType1', 'BsmtFinType2', 'Electrical', 'Functional', 'GarageType', 'SaleType', 'GarageYrBlt', 'KitchenQual',  'BsmtExposure', 'BsmtQual', 'GarageQual', 'GarageCond',  'Utilities', 'GarageFinish', 'BsmtCond']

# columns to impute new category 'none' for
none_imp = ['MiscFeature', 'FireplaceQu', 'PoolQC', 'Alley', 'Fence']

# create a list of quality columns with the same levels
QC_Cols = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC', 'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond', 'PoolQC']

# create a list of corresponding ordinal names for thee columns
QC_Ord_Cols = ['{}_Ord'.format(col) for col in QC_Cols]

# create the mapping dictionary
qc_cols_map_dict = {'Ex':5, 'Gd':4, 'TA':3, 'Fa':2, 'Po':1, 'None':0}

# create the mapping dictionary
lotshape_map_dict = {'Reg':4, 'IR1':3, 'IR2':2, 'IR3':1}

# create the mapping dictionary
landslope_map_dict = {'Gtl':1, 'Mod':2, 'Sev':3}

# create the mapping dictionary
basement_expo_map_dict = {'Gd':4, 'Av':3, 'Mn':2, 'No':1}

# create the mapping dictionary
garagefinish_map_dict = {'Fin':3, 'RFn':2, 'Unf':1}

# create the mapping dictionary
paved_drive_map_dict = {'Y':3, 'P':2, 'N':1}

# create the mapping dictionary
alley_map_dict = {'None':0, 'Grvl':1, 'Pave':2}

# create the mapping dictionary
utilities_map_dict = {'ELO':1, 'NoSeWa':2, 'NoSewr':3, 'AllPub':4}

# create dummy columns list
dummy_cols = ['MSZoning', 'Street', 'LandContour', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'Foundation', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 'CentralAir', 'Electrical', 'Functional', 'GarageType', 'Fence', 'MiscFeature', 'SaleType', 'SaleCondition']