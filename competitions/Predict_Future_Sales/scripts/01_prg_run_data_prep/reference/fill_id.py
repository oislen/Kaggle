# -*- coding: utf-8 -*-
"""
Created on Mon May 10 08:04:30 2021

@author: oislen
"""

import numpy as np

def fill_id(dataset, 
            fill_type, 
            split, 
            fillna = -999
            ):
    
        """
        
        Fill ID Documentation
        
        Function Overview
        
        This function fills in missing index values using either a range of values or a predetermined value
        
        Defaults
        
        fill_id(dataset, 
                fill_type, 
                split, 
                fillna = -999
                )
        
        Parameters
        
        dataset - DataFrame, the data to fill in missing values
        fill_type - String, the type of fill, either 'range' or 'value'
        split - String, the column to split the data on
        fillna - Numeric, the default value to fill missings with if fill type is value
        
        Returns
        
        data - DataFrame, the data with filled in missing indices
        
        Example
        
        fill_id(dataset = join_df, 
                fill_type = 'range', 
                split = 'train'
                )
        
        """
        
        # create a deep copy of the data
        data = dataset.copy(True)
        
        # extract out the data split category
        filt_split = data['data_split'] == split
        
        # determine the number of rows in the split
        nrows = filt_split.sum()
        
        # if fill type is range
        if fill_type == 'range':
            
            # fill in the missing id with a range of values
            data.loc[filt_split, 'ID'] = np.arange(nrows)
            
        # if fill type is value
        elif fill_type == 'value':
            
            # fill in the missing ids with a predetermined value
            data.loc[filt_split, 'ID'] = data.loc[filt_split, 'ID'].fillna(fillna)
            
        return data