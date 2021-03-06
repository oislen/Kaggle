# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 16:51:39 2021

@author: oislen
"""

# import relevant libraries
import pandas as pd
import cons
from scipy.stats import skew
from derive_variables import derive_variables
from tree_feat_imp import tree_feat_imp
from boxcox_trans import boxcox_trans
from standardise_variables import standardise_variables

def engin_base_data(clean_data_fpath,
                    engin_data_fpath,
                    top_n_int_terms = 25,
                    outliers = True,
                    interterms = True,
                    feat_sel = True,
                    boxcox = True,
                    stand = True
                    ):
    
    """
    
    Engineer Base Data Documentation
    
    Function Overview
    
    This functoin generates the engineered base dataset.
    Engineering steps include:
        * Removing outliers
        * Dropping Sales Price
        * Generating interaction terms
        * Performing feature importance on all terms
        * Transforming skewed attributes with boxcox transformation
        * Standardising attributes using robust standardisation method
    
    Defaults
    
    engin_base_data(clean_data_fpath,
                    engin_data_fpath,
                    top_n_int_terms = 25,
                    outliers = True,
                    interterms = True,
                    feat_sel = True,
                    boxcox = True,
                    stand = True
                    )
    
    Parameters
    
    clean_data_fpath - String, the full file path to the input cleaned dataset
    engin_data_fpath - String, the full file path to output the engineered dataset
    top_n_int_terms - Integer, the number of interaction terms to derive, default is 25
    outliers - Boolean, whether or not to remove the outlier observations, default is True
    interterms - Boolean, whether or not to generate interaction terms, default is True
    feat_sel - Boolean, whether or not to perform feature selection, default is True
    boxcox - Boolean, whether or not to transformed highly skewed attributes using boxcox power transformation, defualt is True
    stand - Boolean, whether or not to standardise all features using robust standardisation method, default is True
    
    Returns
    
    0 for successful execution
    
    Example
    
    engin_base_data(clean_data_fpath = 'C:\\Users\\...\\data\\clean.csv',
                    engin_data_fpath = 'C:\\Users\\...\\data\\engin.csv',
                    top_n_int_terms = 25,
                    outliers = True,
                    interterms = True,
                    feat_sel = True,
                    boxcox = True,
                    stand = True
                    )
    
    """
    
    print('Loading clean base data ...')
    
    # load in data
    clean = pd.read_csv(clean_data_fpath, sep = cons.sep)
    
    # if removing outliers
    if outliers == True:
        
        print('Remove outliers ...')
        
        # iterate through the outlier threshold dictionary
        for col, thresh in cons.outlier_dict.items():
            
            # extract out training data
            train_df = clean[clean['Dataset'] == 'train'].copy(deep = True)
        
            # apply outlier filter and extract index
            filt = (clean[col] > thresh)
            index = train_df[filt].index
        
            # apply the full dataset
            clean = clean.drop(index = index)
    
    # reset the index given the recently removed data
    clean = clean.reset_index(drop = 'True')
    
    print('Drop sale price ...')
    
    # drop SalePrice
    clean = clean.drop(columns = 'SalePrice')
    
    # if derviving interaction terms
    if interterms == True:
    
        print('Deriving interaction terms ...')
        
        # extract out the base columns
        clean_cols = clean.columns.tolist()
        
        # extract out the integer columns
        attr_cols = [col for col in clean_cols if col not in cons.tar_cols]
        
        # create interaction terms
        int_data = derive_variables(dataset = clean,
                                    attr = attr_cols,
                                    var_type = 'interaction'
                                    )
        
        # create the concatenation object list
        concat_objs = [clean[clean_cols], int_data]
        
        # create the engineered data by concatenating the base data with the interaction data
        engin = pd.concat(objs = concat_objs, axis = 1)

    # if running feature selection
    if feat_sel == True:
            
        print('Performing feature importance on all terms ...')
        
        # create a list of all predictors
        pred_cols = attr_cols + int_data.columns.tolist()
        
        # extract training data
        y_train = engin.loc[engin['Dataset'] == 'train', cons.y_col]
        X_train = engin.loc[engin['Dataset'] == 'train', pred_cols]
        
        # create a tree model
        model = cons.rfr_mod
        
        # determine the feature importance
        feat_imp = tree_feat_imp(model = model,
                                 y_train = y_train,
                                 X_train = X_train
                                 )
        
        # consider only interaction terms
        int_feat_imp_filt = pd.Series(feat_imp.index).str.contains('_x_').tolist()
        
        # filter out non interaction terms from feat_imp
        feat_imp_sub = feat_imp.loc[int_feat_imp_filt, :].reset_index()
        
        # extract out the important features 
        top_int_feat = feat_imp_sub['Predictor'].head(top_n_int_terms).tolist()
        
        # add in additional variables to enable the interaction effects
        out_vars = clean_cols + top_int_feat
        
        # create the final dataset
        final_data = engin[out_vars]
    
    # if running boxcox transformation
    if boxcox == True:
    
        print('Transforming skewed attributes ...')
    
        # create a list of the numeric columns
        num_cols = final_data.columns[(final_data.dtypes == 'int64') | (final_data.dtypes == 'float64')]
        num_cols = num_cols.drop(['Id', 'logSalePrice'])
        
        # remove columns with less than 20 values
        high_card_num_cols_filt = final_data[num_cols].apply(lambda x: x.nunique(), axis = 0) > 25
        high_card_num_cols = final_data[num_cols].columns[high_card_num_cols_filt]
        
        # determine highly skewed data
        skewed_feats = final_data[high_card_num_cols].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
        skewness = pd.DataFrame({'Skew' :skewed_feats})
        
        # drop unneccessary columns
        skewness = skewness[abs(skewness) > 0.75].dropna()
        bin_cols = skewness.index.astype(str).str.extract(pat = '(^\w+_bin$)', expand = False).dropna().tolist()
        skewness = skewness.drop(index = bin_cols)
        skewed_cols = skewness.index.tolist()
            
        # run boxcox power transformation to remove attribute skew
        boxcox_data = boxcox_trans(dataset = final_data,
                                   attr = skewed_cols
                                   )    
            
        # update the columns
        final_data[skewed_cols] = boxcox_data[skewed_cols]
    
    # if standardising variables
    if stand == True:
        
        print('Standardising data ...')
        
        # create a list of variables to standardise
        stand_cols = final_data.columns.drop(cons.ref_cols).tolist()
        
        # standardise the dataset using the robust scalar method
        final_data[stand_cols] = standardise_variables(dataset = final_data,
                                                       attr = stand_cols,
                                                       stand_type = 'robust'
                                                       )
        
    print('Outputting engineered data ...')
    
    # save output data
    final_data.to_csv(engin_data_fpath,
                      sep = cons.sep,
                      encoding = cons.encoding,
                      header = cons.header,
                      index = cons.index
                      )

        
    return 0
