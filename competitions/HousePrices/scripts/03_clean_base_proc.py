# -*- coding: utf-8 -*-
"""
Created on Sun Nov 18 15:33:13 2018

@author: oislen
"""

# load in relevant libraries
import pandas as pd
import numpy as np
from scipy.stats import skew
import value_analysis as va
import cons

print('Loading input data ...')

# load in data
base = pd.read_csv(cons.base_data_fpath, sep = cons.sep)

# set the id, dataset and target columns
clean = base.copy()

print('Fill in NaN values for base data...')
    
# fill in zeros
clean[cons.zero_imp] = clean[cons.zero_imp].fillna(0)    

# fill in medians
clean[cons.median_imp] = clean[cons.median_imp].apply(lambda x: x.fillna(x.median()), axis = 0) 

# fill in the mode
clean[cons.mode_imp] = clean[cons.mode_imp].apply(lambda x: x.fillna(x.value_counts().index[0], axis = 0))

# fill in the none category
clean[cons.none_imp] = clean[cons.none_imp].fillna('None')       

print('Engineering new features ...')

# create logSalePrice (Will be target)
clean['logSalePrice'] = np.log1p(clean['SalePrice'])

# create the total surface floor attribute
clean['TotalSF'] = clean['TotalBsmtSF'] + clean['1stFlrSF'] + clean['2ndFlrSF']

print('Creating ordinal varibles ...')

# map the categories to ordinal variables
clean[cons.QC_Ord_Cols] = clean[cons.QC_Cols].apply(lambda x: x.map(cons.qc_cols_map_dict), axis = 0)

# map the categories to ordinal variables
clean['LotShape_ord'] = clean['LotShape'].map(cons.lotshape_map_dict)

# map the categories to ordinal variables
clean['LandSlope_ord'] = clean['LandSlope'].map(cons.landslope_map_dict)

# map the categories to ordinal variables
clean['BsmtExposure_ord'] = clean['BsmtExposure'].map(cons.basement_expo_map_dict)

# map the categories to ordinal variables
clean['GarageFinish_ord'] = clean['GarageFinish'].map(cons.garagefinish_map_dict)

# map the categories to ordinal variables
clean['PavedDrive_ord'] = clean['PavedDrive'].map(cons.paved_drive_map_dict)

# map the categories to ordinal variables
clean['Alley_ord'] = clean['Alley'].astype(str).map(cons.alley_map_dict)

# map the categories to ordinal variables
clean['Utilities_ord'] = clean['Utilities'].map(cons.utilities_map_dict)

print('Dummy encoding categorical variables ...')

# create dummy indicators
dummy = pd.get_dummies(data = clean[cons.dummy_cols])

# create concatenation list
concat_list = [clean, dummy]

# concatenate dummies to data
clean = pd.concat(objs = concat_list, axis = 1)

# if removing the numeric outliers from the data
if False:
    
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

if False:
    
    print('Creating interaction terms ...')

    # load in the feature importance data
    feat_imp = pd.read_csv(cons.glm_feat_imp_fpath,
                           sep = '|'
                           )
    
    # create interaction terms from the attributes with high correction (> 0.4)
    der_cols = feat_imp.sort_values(by = ['Test Stat'], ascending = False).Predictor[:30].tolist()

    # create the interaction terms
    int_data = va.derive_variables(dataset = clean,
                                   attr = der_cols,
                                   var_type = 'interaction',
                                   suffix = '_int'
                                   )
    
    # concatenate the interaction data with the clean
    clean = pd.concat(objs = [clean, int_data],
                      sort = False,
                      axis = 1
                      )

if False:
    
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

print('Outputting base file ...')

# output the dataset
clean.to_csv(cons.clean_data_fpath,
             sep = cons.sep,
             encoding = cons.encoding,
             header = cons.header,
             index = cons.index
             )
