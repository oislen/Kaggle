# -*- coding: utf-8 -*-
"""
Created on Sun Nov 18 15:33:13 2018

@author: oislen
"""

#~~~~~ User Inputs ~~~~~

fill_nan = True
map_numeric = True
feature_eng = True
ordinal_rank = True
dummy_encode = True
remove_outliers = True
derive_interact = True
boxcox_trans = False
output_data = True

#~~~~~~~~~~~~~~~~~~~~~~~

"""
#####################
#-- Preliminaries --#
#####################
"""

# load in relevant libraries
import pandas as pd
from scipy.stats import skew

# load cusotm functions
import sys
va_dir = 'C:/Users/User/Documents/Data_Analytics/Python/value_analysis'
sys.path.append(va_dir)
import value_analysis as va

# load in data
input_dir = 'C:\\Users\\User\\Documents\\GitHub\\Kaggle\\HousePrices\\data\\'
base_name = 'base.csv'
base = pd.read_csv(input_dir + base_name, sep = '|')

"""
##################################
#-- Create the Cleaned Dataset --#
##################################
"""

# create the clean dataset
clean = pd.DataFrame()

# set the id, dataset and target columns
tar_cols = ['Id', 'Dataset', 'SalePrice', 'logSalePrice']
clean[tar_cols] = base[tar_cols]

"""
#######################
#-- Fill NaN Values --#
#######################
"""

# if filling in the NaN Values
if fill_nan == True:
    
    print('Fill in NaN values for base data...')
    
    # columns to fill in zeros for
    zero_imp = ['MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF',
                'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath',
                'GarageCars', 'GarageArea'
                ]
    
    # for each zero_imp column
    for col in zero_imp:
        
        # fill in zeros
        base[col] = base[col].fillna(0)    
    
    # columns to fill in medians for
    median_imp = ['LotFrontage',]
    
    # for each median_imp column
    for col in median_imp:
        
        # fill in medians
        base[col] = base[col].fillna(base[col].median())      

    # columns to impute mode for
    mode_imp = ['MSZoning', 'Exterior1st', 'Exterior2nd', 'MasVnrType',
                'BsmtFinType1', 'BsmtFinType2', 'Electrical', 'Functional',
                'GarageType', 'SaleType', 'GarageYrBlt', 'KitchenQual', 
                'BsmtExposure', 'BsmtQual', 'GarageQual', 'GarageCond', 
                'Utilities', 'GarageFinish', 'BsmtCond'
                ]
    
    # for each mode_imp column
    for col in mode_imp:
        
        # fill in the mode
        base[col] = base[col].fillna(base[col].value_counts().index[0])

    # columns to impute new category 'none' for
    none_imp = ['MiscFeature', 'FireplaceQu', 'PoolQC', 'Alley', 'Fence']
    
    # for each none_imp column
    for col in none_imp:
        
        # fill in the none category
        base[col] = base[col].fillna('None')       
        
"""
########################
#-- Transfer Numeric --#
########################
"""

if map_numeric == True:
    
    print('Transfer numeric variales ...')
    
    # create a list of the numeric columns
    num_cols = base.columns[(base.dtypes == 'int64') | (base.dtypes == 'float64')]
    num_cols = num_cols.drop(['Id', 'SalePrice', 'logSalePrice'])
    
    # set the numeric columns
    clean[num_cols] = base[num_cols]

"""
###########################
#-- Feature Engineering --#
###########################
"""

# if creating new features from existing ones
if feature_eng == True:
    
    # create the total surface floor attribute
    clean['TotalSF'] = clean['TotalBsmtSF'] + clean['1stFlrSF'] + clean['2ndFlrSF']

"""
#########################################
#-- Map Ordinal Categorical Varibales --#
#########################################
"""

# if converting nominal categories to ordinal categories
if ordinal_rank == True:
    
    print('Map ordinal categorical variables ...')
    
    ###########################
    #-- Quality / Condition --#
    ###########################
    
    # create a list of quality columns with the same levels
    QC_Cols = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC', 
               'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond', 
               'PoolQC'
               ]
    
    # create a list of corresponding ordinal names for thee columns
    QC_Ord_Cols = ['ExterQual_ord', 'ExterCond_ord', 'BsmtQual_ord', 
                   'BsmtCond_ord', 'HeatingQC_ord', 'KitchenQual_ord',
                   'FireplaceQu_ord', 'GarageQual_ord', 'GarageCond_ord',
                   'PoolQC_ord'
                   ]
    
    # create the mapping dictionary
    map_dict = {'Ex':5, 'Gd':4, 'TA':3, 'Fa':2, 'Po':1, 'None':0}
    
    # for each column
    for idx in range(len(QC_Cols)):
        
        # map the categories to ordinal variables
        clean[QC_Ord_Cols[idx]] = base[QC_Cols[idx]].map(map_dict)
    
    ################
    #-- LotShape --#
    ################
    
    # create the mapping dictionary
    map_dict = {'Reg':4, 'IR1':3, 'IR2':2, 'IR3':1}
    
    # map the categories to ordinal variables
    clean['LotShape_ord'] = base['LotShape'].map(map_dict)
    
    #################
    #-- LandSlope --#
    #################
    
    # create the mapping dictionary
    map_dict = {'Gtl':1, 'Mod':2, 'Sev':3}
    
    # map the categories to ordinal variables
    clean['LandSlope_ord'] = base['LandSlope'].map(map_dict)
    
    ####################
    #-- BsmtExposure --#
    ####################
    
    # create the mapping dictionary
    map_dict = {'Gd':4, 'Av':3, 'Mn':2, 'No':1}
    
    # map the categories to ordinal variables
    clean['BsmtExposure_ord'] = base['BsmtExposure'].map(map_dict)
    
    ####################
    #-- BsmtFinType1 --#
    ####################
    
    #clean['BsmtFinType1_Ord'] = base['BsmtFinType1'].map({})
    
    ####################
    #-- GarageFinish --#
    ####################
    
    # create the mapping dictionary
    map_dict = {'Fin':3, 'RFn':2, 'Unf':1}
    
    # map the categories to ordinal variables
    clean['GarageFinish_ord'] = base['GarageFinish'].map(map_dict)
    
    ##################
    #-- PavedDrive --#
    ##################
    
    # create the mapping dictionary
    map_dict = {'Y':3, 'P':2, 'N':1}
    
    # map the categories to ordinal variables
    clean['PavedDrive_ord'] = base['PavedDrive'].map(map_dict)
    
    #############
    #-- Alley --#
    #############
    
    # create the mapping dictionary
    map_dict = {'None':0, 'Grvl':1, 'Pave':2}
    
    # map the categories to ordinal variables
    clean['Alley_ord'] = base['Alley'].astype(str).map(map_dict)
    
    #################
    #-- Utilities --#
    #################
    
    # create the mapping dictionary
    map_dict = {'ELO':1, 'NoSeWa':2, 'NoSewr':3, 'AllPub':4}
    
    # map the categories to ordinal variables
    clean['Utilities_ord'] = base['Utilities'].map(map_dict)

"""
#################################################
#-- Dummy Encode Nomial Categorical Variables --#
#################################################
"""

# if dummy encoding the categorical variables
if dummy_encode == True:
    
    print('Dummy encoding nominal categorical variables ...')
    
    # define the columns to derive dummy variables for
    dum_cols = base.columns[base.dtypes == 'object'].drop('Dataset')
    
    # ordinal variables to drop
    ord_cols = QC_Cols + ['LotShape', 'BsmtExposure', 'GarageFinish', 
                          'PavedDrive', 'Alley', 'Utilities'
                          ]
    
    # drop the ordinal variables
    dum_cols = dum_cols.drop(ord_cols)
    
    # create the dummy variables
    dummy = va.derive_variables(dataset = base,
                                attr = dum_cols,
                                var_type = 'dummy',
                                suffix = '_bin'
                                )
    
    # append the dummy variables to the dataset
    clean = pd.concat(objs = [clean,
                              dummy], 
                      axis = 1
                      )

"""
#######################
#-- Remove Outliers --#
#######################
"""

# if removing the numeric outliers from the data
if remove_outliers == True:
    
    print('Remove outliers ...')
    
    # remove the outliers
    outliers_1stFlrSF = pd.Series(clean[(clean['1stFlrSF'] > 4000) & (clean['Dataset'] == 'train')].index)
    outliers_BsmtFinSF1 = pd.Series(clean[(clean['BsmtFinSF1'] > 4000) & (clean['Dataset'] == 'train')].index)
    outliers_GrLivArea = pd.Series(clean[(clean['GrLivArea'] > 5000) & (clean['Dataset'] == 'train')].index)
    outliers_LotFrontage = pd.Series(clean[(clean['LotFrontage'] > 300) & (clean['Dataset'] == 'train')].index)
    outliers_MiscVal = pd.Series(clean[(clean['MiscVal'] > 8000) & (clean['Dataset'] == 'train')].index)
    outliers_OpenPorchSF = pd.Series(clean[(clean['OpenPorchSF'] > 500) & (clean['Dataset'] == 'train')].index)
    outliers_TotalBsmtSF = pd.Series(clean[(clean['TotalBsmtSF'] > 6000) & (clean['Dataset'] == 'train')].index)
    outliers_SalePrice = pd.Series(clean[(clean['SalePrice'] > 300000) & (clean['Dataset'] == 'train')].index)
    
    # concatenate outliers
    objs = [outliers_1stFlrSF, outliers_BsmtFinSF1, outliers_GrLivArea, outliers_LotFrontage, outliers_MiscVal, outliers_OpenPorchSF, outliers_TotalBsmtSF, outliers_SalePrice]
    outliers_all = pd.concat(objs = objs, axis = 0).reset_index(drop = True).unique()
    
    # remove outliers 7
    clean = clean.drop(index = outliers_all)
    
    # drop SalePrice
    clean = clean.drop(columns = 'SalePrice')

"""
#########################
#-- Interaction Terms --#
#########################
"""

if derive_interact == True:
    
    print('Creating interaction terms ...')
    
    feat_imp_dir = 'C:\\Users\\User\\Documents\\GitHub\\Kaggle\\HousePrices\\report\\clean_data\\feat_imp'
    
    # load in the feature importance data
    feat_imp = pd.read_csv(feat_imp_dir + '\\GLM_feat_imp.csv',
                           sep = '|'
                           )
    
    # create interaction terms from the attributes with high correction (> 0.4)
    der_cols = feat_imp.sort_values(by = ['Test Stat'], ascending = False).Predictor[:30].tolist()

    # create the interaction terms
    int_data = va.derive_variables(dataset = clean,
                                   attr = der_cols,
                                   var_type = 'interaction',
                                   suffix = '_intr'
                                   )
    
    # concatenate the interaction data with the clean
    clean = pd.concat(objs = [clean, int_data],
                      sort = False,
                      axis = 1
                      )

"""
#############################
#-- BoxCox Transformation --#
#############################
"""

if boxcox_trans == True:
    
    print('Transforming skewed attributes ...')
    
    # create a list of the numeric columns
    num_cols = clean.columns[(clean.dtypes == 'int64') | (clean.dtypes == 'float64')]
    num_cols = num_cols.drop('Id')
    
    # determine highly skewed data
    skewed_feats = clean[num_cols].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
    skewness = pd.DataFrame({'Skew' :skewed_feats})
    
    # drop unneccessary columns
    skewness = skewness[abs(skewness) > 0.75].dropna()
    bin_cols = skewness.index.astype(str).str.extract(pat = '(^\w+_bin$)', expand = False).dropna().tolist()
    skewness = skewness.drop(index = bin_cols)
    skewed_cols = skewness.index.tolist()
    
    # create the interaction terms
    boxcox_data = va.derive_variables(dataset = clean,
                                      attr = skewed_cols,
                                      var_type = 'boxcox'
                                      )
    
    # update the columns
    clean[skewed_cols] = boxcox_data[skewed_cols]

"""
########################
#-- Output Base File --#
########################
"""

if output_data == True:

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
