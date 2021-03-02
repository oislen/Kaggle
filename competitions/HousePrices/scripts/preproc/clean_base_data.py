# -*- coding: utf-8 -*-
"""
Created on Sun Nov 18 15:33:13 2018

@author: oislen
"""

# load in relevant libraries
import cons
import pandas as pd
import numpy as np

def clean_base_data(base_data_fpath,
                    clean_data_fpath
                    ):
    
    """
    
    Clean Base Data Documentation
    
    Function Overview
    
    This funciton cleans and processes the base data for modelling.
    Cleaning steps include:
        * Filling in NaN values
        * Engineering new features e.g. TotalSF, logSalePrice
        * Creating ordinal variables
        * Dummy Encoding categorical variables 
    
    Defaults
    
    clean_base_data(base_data_fpath,
                    clean_data_fpath
                    )
    
    Parameters
    
    base_data_fpath - String, the full file path to the input base dataset
    clean_data_fpath - String, the full file path to output the cleaned dataset
    
    Returns
    
    0 for successful execution
    
    Example
    
    clean_base_data(base_data_fpath = 'C:\\Users\\...\\data\\base.csv',
                    clean_data_fpath = 'C:\\Users\\...\\data\\clean.csv'
                    )
    
    """
    
    print('Loading base data ...')
    
    # load in data
    base = pd.read_csv(base_data_fpath, sep = cons.sep)
    
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
    clean['Alley_ord'] = clean['Alley'].astype(np.str).map(cons.alley_map_dict)
    
    # map the categories to ordinal variables
    clean['Utilities_ord'] = clean['Utilities'].map(cons.utilities_map_dict)
    
    print('Dummy encoding categorical variables ...')
    
    # create dummy indicators
    dummy = pd.get_dummies(data = clean[cons.dummy_cols])
    
    # create concatenation list
    concat_list = [clean, dummy]
    
    # concatenate dummies to data
    clean = pd.concat(objs = concat_list, axis = 1)
    
    print('Extract cleaned features ...')
    
    # extract the clean feature data types
    clean_dtypes = clean.drop(columns = cons.tar_cols).dtypes
    
    # create a filter for string data types
    str_dtyps = clean_dtypes =='object'
    
    # filter out the clean numeric predictor columns
    num_cols = clean_dtypes[~str_dtyps].index.tolist()
    
    # create a list of columns to subset
    sub_cols = cons.tar_cols + num_cols
    
    # subset the cleaned features and target features
    clean = clean[sub_cols]
    
    print('Outputting base file ...')
    
    # output the dataset
    clean.to_csv(clean_data_fpath,
                 sep = cons.sep,
                 encoding = cons.encoding,
                 header = cons.header,
                 index = cons.index
                 )
    
    return 0