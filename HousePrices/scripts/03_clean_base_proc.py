# -*- coding: utf-8 -*-
"""
Created on Sun Nov 18 15:33:13 2018

@author: oislen
"""

"""
#####################
#-- Preliminaries --#
#####################
"""

# load in relevant libraries
import pandas as pd

# load in data
input_dir = 'C:\\Users\\User\\Documents\\Kaggle\\HousePrices\\data\\'
base_name = 'base.csv'
base = pd.read_csv(input_dir + base_name, sep = '|')

# create the clean dataset
clean = pd.DataFrame()

"""
########################
#-- Transfer Numeric --#
########################
"""

print('Transfer numeric variales ...')

num_cols = ['Id', 'Dataset', 'logSalePrice', 'MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual',
            'OverallCond', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea',
            'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF',
            '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea',
            'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath',
            'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces',
            'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF',
            'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch',
            'PoolArea', 'MiscVal', 'MoSold', 'YrSold']

clean[num_cols] = base[num_cols]

"""
#########################################
#-- Map Ordinal Categorical Varibales --#
#########################################
"""

print('Map ordinal categorical variables ...')

###########################
#-- Quality / Condition --#
###########################

QC_Cols = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC', 
           'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond', 'PoolQC'
           ]

QC_Ord_Cols = ['ExterQual_Ord', 'ExterCond_Ord', 'BsmtQual_Ord', 
               'BsmtCond_Ord', 'HeatingQC_Ord', 'KitchenQual_Ord',
               'FireplaceQu_Ord', 'GarageQual_Ord', 'GarageCond_Ord',
               'PoolQC_Ord'
               ]

map_dict = {'Ex':5, 'Gd':4, 'TA':3, 'Fa':2, 'Po':1}

for idx in range(len(QC_Cols)):
    
    clean[QC_Ord_Cols[idx]] = base[QC_Cols[idx]].map(map_dict)

################
#-- LotShape --#
################

map_dict = {'Regular':4, 'IR1':3, 'IR2':2, 'IR3':1}

clean['LotShape_Ord'] = base['LotShape'].map(map_dict)

#################
#-- LandSlope --#
#################

map_dict = {'Gtl':1, 'Mod':2, 'Sev':3}

clean['LandSlope_Ord'] = base['LandSlope'].map(map_dict)

####################
#-- BsmtExposure --#
####################

map_dict = {'Gd':4, 'Av':3, 'Mn':2, 'No':1}

clean['BsmtExposure_Ord'] = base['BsmtExposure'].map(map_dict)

####################
#-- BsmtFinType1 --#
####################

#clean['BsmtFinType1_Ord'] = base['BsmtFinType1'].map({})

####################
#-- GarageFinish --#
####################

map_dict = {'Fin':3, 'RFn':2, 'Unf':1}

clean['GarageFinish_Ord'] = base['GarageFinish'].map(map_dict)

##################
#-- PavedDrive --#
##################

map_dict = {'Y':3, 'P':2, 'N':1}

clean['PavedDrive_Ord'] = base['PavedDrive'].map(map_dict)

#############
#-- Alley --#
#############

map_dict = {'nan':0, 'Gravel':1, 'Pave':2}

clean['Alley_Ord'] = base['Alley'].astype(str).map(map_dict)

#################
#-- Utilities --#
#################

map_dict = {'ELO':1, 'NoSeWa':2, 'NoSewr':3, 'AllPub':4}

clean['Utilities_Ord'] = base['Utilities'].map(map_dict)

"""
################################################
#-- Dichotomize Binary Categorical Variables --#
################################################
"""

print('Dichotomising binary categorical variables ...')

# Street
clean['Street_Gravel'] = (base['Street'] == 'Gravel').astype(int)

# CentralAir
clean['CentralAir_Y'] = (base['CentralAir'] == 'Y').astype(int)

"""
#################################################
#-- Dummy Encode Nomial Categorical Variables --#
#################################################
"""

print('Dummy encoding nominal categorical variables ...')

dum_cols = ['MSZoning', 'LotConfig', 'Neighborhood', 'Condition1', 
            'Condition2', 'RoofStyle', 'RoofMatl', 'Exterior1st', 
            'Exterior2nd', 'MasVnrType', 'Foundation', 'BsmtFinType1', 
            'BsmtFinType2', 'Heating', 'Electrical', 'Functional', 
            'GarageType', 'Fence', 'MiscFeature', 'SaleType', 
            'SaleCondition'
            ]

dummy = pd.get_dummies(data = base[dum_cols])

clean = pd.concat(objs = [clean,
                          dummy], 
                  axis = 1
                  )

"""
#######################
#-- Fill NaN Values --#
#######################
"""

print('fill NaN values ...')

clean['LotFrontage'] = clean['LotFrontage'].fillna(clean['LotFrontage'].median())
clean['MasVnrArea'] = clean['MasVnrArea'].fillna(clean['MasVnrArea'].median())
clean['BsmtFinSF1'] = clean['BsmtFinSF1'].fillna(clean['BsmtFinSF1'].median())
clean['BsmtFinSF2'] = clean['BsmtFinSF2'].fillna(clean['BsmtFinSF2'].median())
clean['BsmtUnfSF'] = clean['BsmtUnfSF'].fillna(clean['BsmtUnfSF'].median())
clean['TotalBsmtSF'] = clean['TotalBsmtSF'].fillna(clean['TotalBsmtSF'].median())
clean['BsmtFullBath'] = clean['BsmtFullBath'].fillna(clean['BsmtFullBath'].median())
clean['BsmtHalfBath'] = clean['BsmtHalfBath'].fillna(clean['BsmtHalfBath'].median())
clean['GarageYrBlt'] = clean['GarageYrBlt'].fillna(clean['GarageYrBlt'].median())
clean['GarageCars'] = clean['GarageCars'].fillna(clean['GarageCars'].median())
clean['GarageArea'] = clean['GarageArea'].fillna(clean['GarageArea'].median())
clean['BsmtQual_Ord'] = clean['BsmtQual_Ord'].fillna(clean['BsmtQual_Ord'].median())
clean['BsmtCond_Ord'] = clean['BsmtCond_Ord'].fillna(clean['BsmtCond_Ord'].median())
clean['KitchenQual_Ord'] = clean['KitchenQual_Ord'].fillna(clean['KitchenQual_Ord'].median())
clean['FireplaceQu_Ord'] = clean['FireplaceQu_Ord'].fillna(clean['FireplaceQu_Ord'].median())
clean['GarageQual_Ord'] = clean['GarageQual_Ord'].fillna(clean['GarageQual_Ord'].median())
clean['GarageCond_Ord'] = clean['GarageCond_Ord'].fillna(clean['GarageCond_Ord'].median())
clean['PoolQC_Ord'] = clean['PoolQC_Ord'].fillna(clean['PoolQC_Ord'].median())
clean['LotShape_Ord'] = clean['LotShape_Ord'].fillna(clean['LotShape_Ord'].median())
clean['BsmtExposure_Ord'] = clean['BsmtExposure_Ord'].fillna(clean['BsmtExposure_Ord'].median())
#clean['BsmtFinType1_Ord'] = clean['BsmtFinType1_Ord'].fillna(clean['BsmtFinType1_Ord'].median())
clean['GarageFinish_Ord'] = clean['GarageFinish_Ord'].fillna(clean['GarageFinish_Ord'].median())
clean['Alley_Ord'] = clean['Alley_Ord'].fillna(clean['Alley_Ord'].median())
clean['Utilities_Ord'] = clean['Utilities_Ord'].fillna(clean['Utilities_Ord'].median())

"""
########################
#-- Output Base File --#
########################
"""

print('Outputting base file ...')

# define the output location and filename
output_filename = 'clean.csv'

# output the dataset
clean.to_csv(input_dir + output_filename,
             sep = '|',
             encoding = 'utf-8',
             header = True,
             index = False
             )
